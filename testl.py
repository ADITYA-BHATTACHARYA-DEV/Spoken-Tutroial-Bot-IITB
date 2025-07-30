from huggingface_hub import get_model_info

info = get_model_info("NousResearch/Nous-Hermes-2-Mistral-7B-DPO")
print(info.pipeline_tag)
