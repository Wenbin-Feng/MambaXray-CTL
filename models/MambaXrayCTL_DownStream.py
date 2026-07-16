import json
import os

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from evalcap.bleu.bleu import Bleu
from evalcap.cider.cider import Cider
from evalcap.rouge.rouge import Rouge
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM, LlamaTokenizer

from arm.Finetuning.models_mamba import arm_base_pz16, arm_large_pz16
from models.contrastive import symmetric_image_text_contrastive_loss


class MambaXrayCTLDownStream(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.save_hyperparameters(args)
        self.json_path = args.annotation
        self.type = args.type
        self._outputs = []

        print(f"Loading vision encoder: {args.vision_model}")
        if args.type == "base":
            self.visual_encoder = arm_base_pz16(self.type)
            print("Loading arm_base_pz16")
        else:
            self.visual_encoder = arm_large_pz16(self.type)
            print("Loading arm_large_pz16")

        checkpoint_model = {}
        if args.vision_model != "None":
            checkpoint = torch.load(args.vision_model, map_location="cpu")
            print(f"Load arm pre-trained checkpoint from: {args.vision_model}")
            checkpoint_model = checkpoint["model"]
            vision_state = {
                key.replace("visual_encoder.", ""): value
                for key, value in checkpoint_model.items()
                if "visual_encoder." in key
            }
            self.visual_encoder.load_state_dict(vision_state)
            self.load_state_dict(checkpoint_model, strict=False)

        if args.vis_use_lora:
            peft_config_visual = LoraConfig(
                r=args.vis_r,
                lora_alpha=args.vis_alpha,
                target_modules=["query", "value"],
                lora_dropout=args.lora_dropout,
                bias="none",
                modules_to_save=["classifier"],
            )
            self.visual_encoder = get_peft_model(self.visual_encoder, peft_config_visual)
            self.visual_encoder.print_trainable_parameters()
            print("Loading vision encoder with LoRA -- Done")
        elif args.freeze_vm:
            for param in self.visual_encoder.parameters():
                param.requires_grad = False
            print(f"Loading Frozen vision encoder: {args.vision_model} -- Done")
        else:
            print(f"Loading Trainable vision encoder: {args.vision_model} -- Done")

        print("Loading LLM ...")
        if "iu" in args.dataset:
            llama_model = args.qwen_model_path
            self.llama_tokenizer = AutoTokenizer.from_pretrained(
                llama_model,
                trust_remote_code=True,
                use_fast=False,
            )
            self.llama_tokenizer.pad_token_id = 0
            self.llama_tokenizer.bos_token_id = 0
            self.llama_model = AutoModelForCausalLM.from_pretrained(
                llama_model,
                torch_dtype=torch.float16,
            )
            self.embed_tokens = self.llama_model.get_input_embeddings()
            for param in self.llama_model.parameters():
                param.requires_grad = False
            print("Loading QWEN Done")
        else:
            llama_model = args.llama_model_path
            self.llama_tokenizer = LlamaTokenizer.from_pretrained(
                llama_model,
                trust_remote_code=False,
                use_fast=False,
            )
            self.llama_tokenizer.pad_token_id = 0
            if args.low_resource:
                self.llama_model = LlamaForCausalLM.from_pretrained(
                    llama_model,
                    torch_dtype=torch.float16,
                    load_in_8bit=True,
                    device_map="auto",
                )
            else:
                self.llama_model = LlamaForCausalLM.from_pretrained(
                    llama_model,
                    torch_dtype=torch.float16,
                )

            self.embed_tokens = self.llama_model.get_input_embeddings()
            if args.llm_use_lora:
                peft_config = LoraConfig(
                    task_type=TaskType.CAUSAL_LM,
                    inference_mode=False,
                    r=args.llm_r,
                    lora_alpha=args.llm_alpha,
                    lora_dropout=args.lora_dropout,
                    target_modules=["q_proj", "v_proj"],
                )
                self.llama_model = get_peft_model(self.llama_model, peft_config)
                self.llama_model.print_trainable_parameters()
                llama_state = {
                    key.replace("text_encoder.", ""): value
                    for key, value in checkpoint_model.items()
                    if "text_encoder." in key
                }
                self.llama_model.load_state_dict(llama_state, strict=False)
                print("Loading LLAMA LoRA Done")
            else:
                print("Loading LLAMA Done")
            for param in self.llama_model.parameters():
                param.requires_grad = False

        self.llama_proj = nn.Linear(
            self.visual_encoder.num_features,
            self.llama_model.config.hidden_size,
        )
        self.layer_norm = nn.LayerNorm(self.llama_model.config.hidden_size)
        self.text_proj = nn.Linear(
            self.llama_model.config.hidden_size,
            self.llama_model.config.hidden_size,
        )
        self.end_sym = args.end_sym
        self.prompt = "Generate a comprehensive and detailed diagnosis report for this chest xray image."
        self.val_step_outputs = []
        self.val_score = 0.0

        if args.delta_file is not None:
            state_dict = torch.load(
                args.delta_file,
                map_location=torch.device(f"cuda:{torch.cuda.current_device()}"),
            )["model"]
            self.load_state_dict(state_dict=state_dict, strict=False)
            print(f"Load checkpoint from {args.delta_file}")

    def contrastive_loss(self, image_embeddings, text_embeddings, temperature=None):
        if temperature is None:
            temperature = self.hparams.contrastive_temperature
        return symmetric_image_text_contrastive_loss(
            image_embeddings,
            text_embeddings,
            temperature=temperature,
        )

    def score(self, ref, hypo):
        scorers = [
            (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
            (Rouge(), "ROUGE_L"),
            (Cider(), "CIDEr"),
        ]
        final_scores = {}
        if self.args.dataset == "chinese":
            hypo = {key: [" ".join(value) for value in values] for key, values in hypo.items()}
            ref = {key: [" ".join(value) for value in values] for key, values in ref.items()}
        for scorer, method in scorers:
            score, _ = scorer.compute_score(ref, hypo)
            if isinstance(score, list):
                for metric, value in zip(method, score):
                    final_scores[metric] = value
            else:
                final_scores[method] = score
        return final_scores

    def encode_img(self, images, segmentation=None):
        image_embeds = []
        for image in images:
            image_embeds.append(self.visual_encoder(image, segmentation))
        image_embeds = torch.stack(image_embeds).mean(0)
        inputs_llama = self.llama_proj(image_embeds)
        atts_llama = torch.ones(
            inputs_llama.size()[:-1],
            dtype=torch.long,
            device=inputs_llama.device,
        )
        return inputs_llama, atts_llama

    def prompt_wrap(self, img_embeds, atts_img):
        prompt = f"Human: <Img><ImageHere></Img> {self.prompt} \nAssistant:"
        batch_size = img_embeds.shape[0]
        p_before, p_after = prompt.split("<ImageHere>")
        p_before_tokens = self.llama_tokenizer(
            p_before,
            return_tensors="pt",
            add_special_tokens=False,
        ).to(img_embeds.device)
        p_after_tokens = self.llama_tokenizer(
            p_after,
            return_tensors="pt",
            add_special_tokens=False,
        ).to(img_embeds.device)
        p_before_embeds = self.embed_tokens(p_before_tokens.input_ids).expand(batch_size, -1, -1)
        p_after_embeds = self.embed_tokens(p_after_tokens.input_ids).expand(batch_size, -1, -1)
        wrapped_img_embeds = torch.cat([p_before_embeds, img_embeds, p_after_embeds], dim=1)
        wrapped_atts_img = atts_img[:, :1].expand(-1, wrapped_img_embeds.shape[1])
        return wrapped_img_embeds, wrapped_atts_img

    def forward(self, samples):
        image = samples["image"]
        img_embeds, atts_img = self.encode_img(image)
        img_embeds = self.layer_norm(img_embeds)
        image_embeddings = img_embeds.mean(dim=1)
        img_embeds, atts_img = self.prompt_wrap(img_embeds, atts_img)

        self.llama_tokenizer.padding_side = "right"
        text = [value + self.end_sym for value in samples["input_text"]]
        to_regress_tokens = self.llama_tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.hparams.max_length,
            add_special_tokens=False,
        ).to(image[0].device)
        targets = to_regress_tokens.input_ids.masked_fill(
            to_regress_tokens.input_ids == 0,
            -100,
        )
        empty_targets = torch.full(
            (atts_img.shape[0], atts_img.shape[1] + 1),
            -100,
            dtype=torch.long,
            device=image[0].device,
        )
        targets = torch.cat([empty_targets, targets], dim=1)

        batch_size = img_embeds.shape[0]
        bos = torch.full(
            (batch_size, 1),
            self.llama_tokenizer.bos_token_id,
            dtype=to_regress_tokens.input_ids.dtype,
            device=to_regress_tokens.input_ids.device,
        )
        bos_embeds = self.embed_tokens(bos)
        atts_bos = atts_img[:, :1]
        to_regress_embeds = self.embed_tokens(to_regress_tokens.input_ids)
        inputs_embeds = torch.cat([bos_embeds, img_embeds, to_regress_embeds], dim=1)
        attention_mask = torch.cat(
            [atts_bos, atts_img, to_regress_tokens.attention_mask],
            dim=1,
        )
        outputs = self.llama_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True,
            labels=targets,
        )
        text_embeddings = self.text_proj(
            self.llama_model.get_input_embeddings()(to_regress_tokens.input_ids).mean(dim=1)
        )
        matching_loss = self.contrastive_loss(image_embeddings, text_embeddings)
        total_loss = outputs.loss + matching_loss * self.hparams.contrastive_loss_weight
        return {"loss": total_loss}

    def training_step(self, batch, batch_idx):
        result = self(batch)
        self.log_dict(result, prog_bar=True)
        return result

    def save_checkpoint(self, eval_res):
        current_epoch = self.trainer.current_epoch
        global_step = self.trainer.global_step
        trainable = {
            key: value.requires_grad
            for key, value in self.named_parameters()
            if value.requires_grad
        }
        state_dict = self.state_dict()
        for key in list(state_dict):
            if key not in trainable:
                del state_dict[key]
        save_obj = {
            "model": state_dict,
            "config": self.hparams,
            "epoch": current_epoch,
            "step": global_step,
        }
        checkpoint_dir = os.path.join(self.hparams.savedmodel_path, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
        filename = "checkpoint_epoch{}_step{}_bleu{:3f}_cider{:3f}.pth".format(
            current_epoch,
            global_step,
            eval_res["Bleu_4"],
            eval_res["CIDEr"],
        )
        save_to = os.path.join(checkpoint_dir, filename)
        self.print(f"Saving checkpoint at step {global_step} to {save_to}.")
        torch.save(save_obj, save_to)

    def validation_step(self, samples, batch_idx):
        self.llama_tokenizer.padding_side = "right"
        to_regress_tokens = self.llama_tokenizer(
            samples["input_text"],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.hparams.max_length,
            add_special_tokens=False,
        )
        image = samples["image"]
        img_embeds, atts_img = self.encode_img(image)
        img_embeds = self.layer_norm(img_embeds)
        img_embeds, atts_img = self.prompt_wrap(img_embeds, atts_img)
        batch_size = img_embeds.shape[0]
        bos = torch.full(
            (batch_size, 1),
            self.llama_tokenizer.bos_token_id,
            dtype=atts_img.dtype,
            device=atts_img.device,
        )
        inputs_embeds = torch.cat([self.embed_tokens(bos), img_embeds], dim=1)
        attention_mask = torch.cat([atts_img[:, :1], atts_img], dim=1)
        outputs = self.llama_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            num_beams=self.hparams.beam_size,
            do_sample=self.hparams.do_sample,
            min_new_tokens=self.hparams.min_new_tokens,
            max_new_tokens=self.hparams.max_new_tokens,
            repetition_penalty=self.hparams.repetition_penalty,
            length_penalty=self.hparams.length_penalty,
            temperature=self.hparams.temperature,
        )
        hypo = [self.decode(tokens) for tokens in outputs]
        ref = [self.decode(tokens) for tokens in to_regress_tokens["input_ids"]]
        self.val_step_outputs.append({"hypo": hypo, "ref": ref, "id": samples["id"]})
        return hypo, ref

    def decode(self, output_token):
        if output_token.numel() and output_token[0].item() == 0:
            output_token = output_token[1:]
        if output_token.numel() and output_token[0].item() == 1:
            output_token = output_token[1:]
        output_text = self.llama_tokenizer.decode(output_token, add_special_tokens=False)
        return output_text.split("</s>")[0].strip().replace("<unk>", "")

    def on_validation_epoch_end(self):
        ref, hypo, ids = [], [], []
        for output in self.val_step_outputs:
            ref.extend(output["ref"])
            hypo.extend(output["hypo"])
            ids.extend(output["id"])
        ref = {key: [value] for key, value in zip(ids, ref)}
        hypo = {key: [value] for key, value in zip(ids, hypo)}
        eval_res = self.score(ref=ref, hypo=hypo)
        self.log_dict(eval_res, sync_dist=True, logger=True)

        result_folder = os.path.join(self.hparams.savedmodel_path, "result")
        os.makedirs(result_folder, exist_ok=True)
        current_epoch = self.trainer.current_epoch
        global_step = self.trainer.global_step
        with open(
            os.path.join(result_folder, f"result_{current_epoch}_{global_step}.json"),
            "w",
            encoding="utf-8",
        ) as stream:
            json.dump(hypo, stream, ensure_ascii=False)
        with open(os.path.join(result_folder, "refs.json"), "w", encoding="utf-8") as stream:
            json.dump(ref, stream, ensure_ascii=False)
        self.print(eval_res)

        val_score = sum(
            eval_res[score_type] * weight
            for score_type, weight in zip(self.hparams.scorer_types, self.hparams.weights)
        )
        if self.trainer.local_rank == 0:
            if val_score > self.val_score:
                self.save_checkpoint(eval_res)
                self.val_score = val_score
            elif val_score > 0.125:
                self.save_checkpoint(eval_res)
        self.val_step_outputs.clear()

    def test_step(self, samples, batch_idx):
        self.llama_tokenizer.padding_side = "right"
        to_regress_tokens = self.llama_tokenizer(
            samples["input_text"],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.hparams.max_length,
            add_special_tokens=False,
        )
        image = samples["image"]
        img_embeds, atts_img = self.encode_img(image)
        img_embeds = self.layer_norm(img_embeds)
        image_embeddings = img_embeds.mean(dim=1)
        img_embeds, atts_img = self.prompt_wrap(img_embeds, atts_img)
        batch_size = img_embeds.shape[0]
        bos = torch.full(
            (batch_size, 1),
            self.llama_tokenizer.bos_token_id,
            dtype=atts_img.dtype,
            device=atts_img.device,
        )
        inputs_embeds = torch.cat([self.embed_tokens(bos), img_embeds], dim=1)
        attention_mask = torch.cat([atts_img[:, :1], atts_img], dim=1)
        outputs = self.llama_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            num_beams=self.hparams.beam_size,
            do_sample=self.hparams.do_sample,
            min_new_tokens=self.hparams.min_new_tokens,
            max_new_tokens=self.hparams.max_new_tokens,
            repetition_penalty=self.hparams.repetition_penalty,
            length_penalty=self.hparams.length_penalty,
            temperature=self.hparams.temperature,
        )
        hypo = [self.decode(tokens) for tokens in outputs]
        ref = [self.decode(tokens) for tokens in to_regress_tokens["input_ids"]]

        with torch.no_grad():
            hypothesis_tokens = self.llama_tokenizer(
                hypo,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.hparams.max_length,
                add_special_tokens=False,
            ).to(image[0].device)
            text_embeddings = self.llama_model.get_input_embeddings()(
                hypothesis_tokens.input_ids
            ).mean(dim=1)
            text_embeddings = F.normalize(text_embeddings, p=2, dim=1)
            image_embeddings = F.normalize(image_embeddings, p=2, dim=1)
            confidence = torch.cosine_similarity(image_embeddings, text_embeddings).tolist()

        result = {
            "hypo": hypo,
            "ref": ref,
            "id": samples["id"],
            "confidence": confidence,
        }
        self._outputs.append(result)
        return result

    def on_test_epoch_end(self):
        ref, hypo, ids = [], [], []
        for output in self._outputs:
            ref.extend(output["ref"])
            hypo.extend(output["hypo"])
            ids.extend(output["id"])
        ref = {key: [value] for key, value in zip(ids, ref)}
        hypo = {key: [value] for key, value in zip(ids, hypo)}
        eval_res = self.score(ref=ref, hypo=hypo)

        result_folder = os.path.join(self.hparams.savedmodel_path, "result")
        os.makedirs(result_folder, exist_ok=True)
        with open(os.path.join(result_folder, "test_result.json"), "w", encoding="utf-8") as stream:
            json.dump(hypo, stream, ensure_ascii=False)
        with open(os.path.join(result_folder, "test_refs.json"), "w", encoding="utf-8") as stream:
            json.dump(ref, stream, ensure_ascii=False)
        self.print(f"Test result of {self.hparams.delta_file}: {eval_res}")
        self._outputs.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer,
            T_max=self.hparams.max_epochs,
            eta_min=1e-6,
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def get_progress_bar_dict(self):
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        return items

    def optimizer_zero_grad(self, epoch, batch_idx, optimizer):
        optimizer.zero_grad()
