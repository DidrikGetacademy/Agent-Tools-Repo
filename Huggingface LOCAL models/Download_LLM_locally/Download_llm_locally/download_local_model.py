from huggingface_hub import snapshot_download
from smolagents import CodeAgent, FinalAnswerTool,TransformersModel,DuckDuckGoSearchTool,FinalAnswerTool

import gc 
import time
import os
import torch
import logging







def download_deepseek_coder_7b_instruct():
    """Advanced code generation and completion.
    Model Inseight---->
    ------------------------------------------------------------------------
    -6.7 billion parameters.
    -Context Length: 16K tokens.
    -Architecture: Transformer-based model optimized for code tasks, featuring a fill-in-the-blank objective for project-level code completion and infilling.
    -Training Data: 2 trillion tokens comprising 87% code and 13% natural language in English and Chinese.
    Model Features ----> 
    ------------------------------------------------------------------------
    -Supports project-level code completion and infilling.
    -Pre-trained on repository-level code corpus.
    
    Ideal Use Cases: 
    ------------------------------------------------------------------------
    -Code generation, completion, and understanding tasks.
    """
    model_repo = "deepseek-ai/deepseek-coder-6.7b-instruct"
    local_dir = "./local_model/deepseek-coder-6.7b-instruct"

    if not os.path.exists(local_dir):
        print(f"📥 Downloading {model_repo} into {local_dir}...")
        snapshot_download(
            repo_id=model_repo,
            local_dir=local_dir,
            resume_download=True,
            local_dir_use_symlinks=False
        )
        print(f"✅ deepseek-coder-6.7b-instruct downloaded successfully.")
    else:
        print(f"✅ Model already exists at {local_dir}, skipping download.")


























#you have to ask for access to this model
def download_Mistral_7B_Instruct_v02():
    """General-purpose language understanding and generation.
    Model Inseight---->
    ------------------------------------------------------------------------
    -Model Size: 7.3 billion parameters.
    -Context Length: 32K tokens.
    -Architecture: Transformer model utilizing Grouped-Query Attention (GQA) and Sliding Window Attention (SWA) mechanisms for efficient long-context processing.
    -Training Data: The base model, Mistral-7B-v0.2, was pre-trained on publicly available datasets. The instruct version was fine-tuned to enhance instruction-following capabilities.
   
    Model Features ----> 
    ------------------------------------------------------------------------
    -Fine-tuned version of Mistral-7B-v0.1 with improved instruction-following capabilities.
    -Enhanced performance in tasks like mathematics, code generation, and reasoning.


    Ideal Use Cases: 
    ------------------------------------------------------------------------
    -Instruction-following tasks, content generation, and general-purpose language tasks.
    
    """
    model_repo = "mistralai/Mistral-7B-Instruct-v0.2"
    local_dir = "./local_model/Mistral-7B-Instruct-v0.2"

    if not os.path.exists(local_dir):
        print(f"📥 Downloading {model_repo} into {local_dir}...")
        snapshot_download(
            repo_id=model_repo,
            local_dir=local_dir,
            resume_download=True,
            local_dir_use_symlinks=False
        )
        print(f"✅ Mistral-7B-Instruct-v0.2 downloaded successfully.")
    else:
        print(f"✅ Model already exists at {local_dir}, skipping download.")

























def download_Mistral_7B_Instruct_v03():
    """
    Model Size	7.3 billion parameters,
    Context Length	32,768 tokens (32K),
    Architecture	Transformer Decoder-only, GPT-style,
    Attention Type	Multi-head with Grouped-Query Attention (GQA),
    Training	Fine-tuned version of mistralai/Mistral-7B-v0.1,
    Use Case	Instruction-tuned for chat, reasoning, Q&A, code, summarization, etc.
    """
    model_repo = "mistralai/Mistral-7B-Instruct-v0.3"
    local_dir = "./local_model/mistralai/Mistral-7B-Instruct-v0.3"

    if not os.path.exists(local_dir):
        print(f"📥 Downloading {model_repo} into {local_dir}...")
        snapshot_download(
            repo_id=model_repo,
            local_dir=local_dir,
            resume_download=True,
            local_dir_use_symlinks=False
        )
        print(f"✅ Mistral-7B-Instruct-v0.2 downloaded successfully.")
    else:
        print(f"✅ Model already exists at {local_dir}, skipping download.")



























def download_Qwen_Coder2_5_Instruct_1_5b():
    """Lightweight model for instruction-following tasks.
    Model Inseight---->
    ------------------------------------------------------------------------
    -Model Size: 1.5 billion parameters
    -Context Length: 32K tokens.
    -Architecture: Transformer architecture with Rotary Positional Embeddings (RoPE), SwiGLU activation, RMSNorm, Attention QKV bias, and tied word embeddings.
    -Training Data: Pre-trained and post-trained on a diverse corpus, including code, mathematical problems, and natural language data. Specific token counts are not disclosed.
    Model Features ----> 
    ------------------------------------------------------------------------
    -Suitable for environments with limited computational resources.
    
    Ideal Use Cases: 
    ------------------------------------------------------------------------
    -Instruction-following tasks in resource-constrained settings
    
    """
    model_repo = "Qwen/Qwen2.5-1.5B-Instruct"
    local_dir = "./local_model/Qwen2.5-1.5B-Instruct"

    if not os.path.exists(local_dir):
        print(f"📥 Downloading {model_repo} into {local_dir}...")
        snapshot_download(
            repo_id=model_repo,
            local_dir=local_dir,
            resume_download=True,
            local_dir_use_symlinks=False
        )
        print(f"✅ Qwen2.5-1.5B-Instruct downloaded successfully.")
    else:
        print(f"✅ Model already exists at {local_dir}, skipping download.")






















def download_Qwen_7b_Instruct_1M():
    """Handling tasks requiring long-context understanding.
    Model Inseight---->
    ------------------------------------------------------------------------
    -Model Size: 7.6 billion parameters.
    -Context Length: Up to 1 million tokens.
    -Architecture: Transformer architecture with RoPE, SwiGLU, RMSNorm, Attention QKV bias, and tied word embeddings.
    -Training Data: Enhanced for long-context understanding through long-context pre-training and post-training techniques. Specific datasets and token counts are not detailed.
    
    Model Features ----> 
    ------------------------------------------------------------------------
    -Optimized for long-context tasks.

    Ideal Use Cases: 
    ------------------------------------------------------------------------
    -Processing long documents, extended conversations, and other long-context scenarios.
    """
    model_repo = "Qwen/Qwen2.5-7B-Instruct-1M"
    local_dir = "./local_model/Qwen2.5-7B-Instruct-1M"

    if not os.path.exists(local_dir):
        print(f"📥 Downloading {model_repo} into {local_dir}...")
        snapshot_download(
            repo_id=model_repo,
            local_dir=local_dir,
            resume_download=True,
            local_dir_use_symlinks=False
        )
        print(f"✅ Qwen2.5-7B-Instruct-1M downloaded successfully.")
    else:
        print(f"✅ Model already exists at {local_dir}, skipping download.")

























def download_Qwen_Coder_Instruct_7b():
    """Code generation and understanding.
    Model Inseight---->
    ------------------------------------------------------------------------
    -Model Size: 7.6 billion parameters.
    -Context Length: 128K tokens.
    -Architecture: Transformer architecture with RoPE, SwiGLU, RMSNorm, Attention QKV bias, and tied word embeddings.
    -Training Data: Trained on 5.5 trillion tokens, including source code, text-code grounding data, and synthetic data.
    Model Features ----> 
    ------------------------------------------------------------------------
    -Instruction-tuned for code-related tasks.
    -Enhanced code reasoning and generation capabilities.
    
    Ideal Use Cases: 
    ------------------------------------------------------------------------
    -Code generation, debugging, and code-related instruction tasks
    """
    model_repo = "Qwen/Qwen2.5-Coder-7B-Instruct"
    local_dir = "./local_model/Qwen/Qwen2.5-Coder-7B-Instruct"

    if not os.path.exists(local_dir):
        print(f"📥 Downloading {model_repo} into {local_dir}...")
        snapshot_download(
            repo_id=model_repo,
            local_dir=local_dir,
            resume_download=True,
            local_dir_use_symlinks=False
        )
        print(f"✅ Qwen2.5-Coder-7B-Instruct downloaded successfully.")
    else:
        print(f"✅ Model already exists at {local_dir}, skipping download.")






















def download_Qwen2_5_Coder_3B_Instruct():
    """ Efficient code generation in resource-constrained environments.
    Model Inseight---->
    ------------------------------------------------------------------------
    -Model Size: 3 billion parameters.
    -Context Length: 32K tokens.
    -Architecture: Transformer architecture with RoPE, SwiGLU, RMSNorm, Attention QKV bias, and tied word embeddings.
    -Training Data: Similar to its 7B counterpart, it was trained on 5.5 trillion tokens encompassing source code, text-code grounding data, and synthetic data. 
    
    Model Features ----> 
    ------------------------------------------------------------------------
    -Instruction-tuned for code-related tasks.
    -Balanced performance and resource usage.

    
    Ideal Use Cases: 
    ------------------------------------------------------------------------
    -Code generation and understanding in environments with limited computational resources
    """
    model_repo = "Qwen/Qwen2.5-Coder-3B-Instruct"
    local_dir = "./local_model/Qwen/Qwen2.5-Coder-3B-Instruct"

    if not os.path.exists(local_dir):
        print(f"📥 Downloading {model_repo} into {local_dir}...")
        snapshot_download(
            repo_id=model_repo,
            local_dir=local_dir,
            resume_download=True,
            local_dir_use_symlinks=False
        )
        print(f"✅ Qwen2.5-Coder-3B-Instruct downloaded successfully.")
    else:
        print(f"✅ Model already exists at {local_dir}, skipping download.")



















def download_microsoft_Phi_3_mini_128k_instruct():
    """Lightweight model for instruction-following tasks with long-context support.
    Model Inseight---->
    ------------------------------------------------------------------------
    -Model Size: 3.8 billion parameters.
    -Context Length: 128K tokens.
    -Architecture: Dense decoder-only Transformer model fine-tuned with Supervised Fine-Tuning (SFT) and Direct Preference Optimization (DPO) to ensure alignment with human preferences and safety guidelines.
    -Training Data: Utilized the Phi-3 datasets, which include synthetic data and filtered publicly available website data, focusing on high-quality and reasoning-dense content. 
   
    Model Features ----> 
    ------------------------------------------------------------------------
    -Trained on synthetic and filtered public data focusing on high-quality, reasoning-dense content.
    -Supports long-context tasks in resource-constrained environments.

    Ideal Use Cases: 
    ------------------------------------------------------------------------
    -Ideal Use Cases: Instruction-following tasks requiring long-context understanding in environments with limited resources

    """
    model_repo = "microsoft/Phi-3-mini-128k-instruct"
    local_dir = "./local_model/microsoft/microsoft/Phi-3-mini-128k-instruct"

    if not os.path.exists(local_dir):
        print(f"📥 Downloading {model_repo} into {local_dir}...")
        snapshot_download(
            repo_id=model_repo,
            local_dir=local_dir,
            resume_download=True,
            local_dir_use_symlinks=False
        )
        print(f"✅ microsoft/Phi-3-mini-128k-instruct downloaded successfully.")
    else:
        print(f"✅ Model already exists at {local_dir}, skipping download.")










def download_microsoft_Phi_4_mini_instruct():
    """ A lightweight, efficient model optimized for environments with limited computational resources.
    Model Inseight---->
    ------------------------------------------------------------------------
    -Model Size: 3.8 billion parameters
    -Context Length: 128K tokens
    -Architecture: Dense decoder-only Transformer
    -Training Data: 5 trillion tokens
     Model Features ----> 
    ------------------------------------------------------------------------
    -Designed for memory and compute-constrained environments.
    -Supports a 128K token context length.
    -Trained on synthetic and filtered public data, focusing on high-quality, reasoning-dense content.
    -Incorporates supervised fine-tuning and direct preference optimization for precise instruction adherence and robust safety measures.

    Ideal Use Cases: 
    ------------------------------------------------------------------------
    -Mobile applications, 
    -edge devices,
    -scenarios where low latency and efficient performance are critical
    
    """
    model_repo = "microsoft/Phi-4-mini-instruct "
    local_dir = "./local_model/microsoft/microsoft/Phi-4-mini-instruct "
    if not os.path.exists(local_dir):
        print(f"📥 Downloading {model_repo} into {local_dir}...")
        snapshot_download(
            repo_id=model_repo,
            local_dir=local_dir,
            resume_download=True,
            local_dir_use_symlinks=False
        )
        print(f"✅ microsoft/microsoft/Phi-4-mini-instruct downloaded successfully.")
    else:
        print(f"✅ Model already exists at {local_dir}, skipping download.")















def download_microsoft_Phi_4_multimodal_instruct():
    """A versatile model capable of processing and understanding multiple modalities, including text, images, and audio.
    Model Inseight---->
    ------------------------------------------------------------------------
    -Model Size: 5.6 billion parameters
    -Context Length: 128K tokens
    -Architecture: Multimodal model integrating text, vision, and speech/audio inputs
    Training Data: 
     5 trillion text tokens
     2.3 million hours of speech data
     1.1 trillion image-text tokens
    
    Model Features ---> Handles text, image, and audio inputs, generating text outputs.
    ------------------------------------------------------------------------
    -Supports a 128K token context length.
    -Text: Supports 24 languages
    -Vision: Primarily English.
    -Audio: English, Chinese, German, French, Italian, Japanese, Spanish, Portuguese.

    
    Ideal Use Cases: 
    ------------------------------------------------------------------------
    Applications requiring multimodal understanding, such as 
    virtual assistants that interpret 
    -images 
    -audio,
    -tools that analyze documents combining text and visuals
    """
    model_repo = "microsoft/Phi-4-multimodal-instruct"
    local_dir = "./local_model/microsoft/microsoft/Phi-4-multimodal-instruct"
    if not os.path.exists(local_dir):
        print(f"📥 Downloading {model_repo} into {local_dir}...")
        snapshot_download(
            repo_id=model_repo,
            local_dir=local_dir,
            resume_download=True,
            local_dir_use_symlinks=False
        )
        print(f"✅ microsoft/Phi-4-multimodal-instruct downloaded successfully.")
    else:
        print(f"✅ Model already exists at {local_dir}, skipping download.")









def download_microsoft_microsoft_Phi_4_reasoning_plus():
    """A model fine-tuned for advanced reasoning tasks, particularly in mathematics, science, and coding.
    Model Inseight---->
    ------------------------------------------------------------------------
    -Model Size:  14 billion parameters
    -Context Length: 32K tokens (extendable up to 64K tokens)
    -Architecture: Dense decoder-only Transformer
    Trainingdata:
    -Supervised Fine-Tuning (SFT): Over 1.4 million prompts and high-quality answers containing long reasoning traces generated using OpenAI's o3-mini model. The datasets cover topics in STEM (science, technology, engineering, and mathematics), coding, and safety-focused tasks
    -Reinforcement Learning (RL): Approximately 6,000 high-quality math-focused problems with verifiable solutions

    Model Features --->
    --------------------------------------------------------------------------------------
    -Built upon the base Phi-4 model with 14 billion parameters.
    -Supports a 32K token context length, extendable up to 64K for complex tasks.
    -Fine-tuned using supervised learning on chain-of-thought datasets and reinforcement learning for enhanced reasoning capabilities.
    -Generates detailed reasoning steps followed by concise summaries.
    
    Ideal Use Cases: 
    --------------------------------------------------------------------------------------
    -Tasks requiring deep analytical thinking, such as complex problem-solving in mathematics, 
    -cientific research
    -advanced coding challenges
    """
    model_repo = "microsoft/Phi-4-reasoning-plus"
    local_dir = "./local_model/microsoft/microsoft/microsoft/Phi-4-reasoning-plus"
    if not os.path.exists(local_dir):
        print(f"📥 Downloading {model_repo} into {local_dir}...")
        snapshot_download(
            repo_id=model_repo,
            local_dir=local_dir,
            resume_download=True,
            local_dir_use_symlinks=False
        )
        print(f"✅ microsoft/Phi-4-reasoning-plus downloaded successfully.")
    else:
        print(f"✅ Model already exists at {local_dir}, skipping download.")








def download_microsoft_microsoft_Phi_4_mini_reasoning():
    """A model fine-tuned for advanced reasoning tasks, particularly in mathematics, science, and coding.
    Model Inseight---->
    ------------------------------------------------------------------------
    -Model Size:  3.84 billion parameters
    -Context Length: 32K tokens (extendable up to 64K tokens)
    -Architecture: Dense decoder-only Transformer

    """
    model_repo = "microsoft/Phi-4-mini-reasoning"
    local_dir = "./local_model/microsoft/microsoft/microsoft/microsoft/Phi-4-mini-reasoning"
    if not os.path.exists(local_dir):
        print(f"📥 Downloading {model_repo} into {local_dir}...")
        snapshot_download(
            repo_id=model_repo,
            local_dir=local_dir,
            resume_download=True,
            local_dir_use_symlinks=False
        )
        print(f"✅ microsoft/Phi-4-reasoning-plus downloaded successfully.")
    else:
        print(f"✅ Model already exists at {local_dir}, skipping download.")





def download_microsoft_microsoft_Phi_4_mini_instruct_onnx():
    """

    """
    model_repo = "microsoft/Phi-4-mini-instruct-onnx"
    local_dir = "./local_model/microsoft/Phi-4-mini-instruct-onnx"
    if not os.path.exists(local_dir):
        print(f"📥 Downloading {model_repo} into {local_dir}...")
        snapshot_download(
            repo_id=model_repo,
            local_dir=local_dir,
            resume_download=True,
            local_dir_use_symlinks=False
        )
        print(f"✅ microsoft/Phi-4-reasoning-plus downloaded successfully.")
    else:
        print(f"✅ Model already exists at {local_dir}, skipping download.")







def download_microsoft_Mixtral_8x7B_instruct_v0_1():
    """A model fine-tuned for advanced reasoning tasks, particularly in mathematics, science, and coding.
    Model Inseight---->
    ------------------------------------------------------------------------
    -Model Size:  14 billion parameters
    -Context Length: 32K tokens (extendable up to 64K tokens)
    -Architecture: Dense decoder-only Transformer

    """
    model_repo = "mistralai/Mixtral-8x7B-Instruct-v0.1"
    local_dir = "./local_model/local_models_path/mistralai/Mixtral-8x7B-Instruct-v0.1"
    if not os.path.exists(local_dir):
        print(f"📥 Downloading {model_repo} into {local_dir}...")
        snapshot_download(
            repo_id=model_repo,
            local_dir=local_dir,
            resume_download=True,
            local_dir_use_symlinks=False
        )
        print(f"✅ mistralai/Mixtral-8x7B-Instruct-v0.1downloaded successfully.")
    else:
        print(f"✅ Model already exists at {local_dir}, skipping download.")







load_in_8bit = False  
def setup_logger(model_id=""):
    import logging
    model_id = model_id.replace("/", "_").replace("\\", "_")

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    if logger.hasHandlers():
        logger.handlers.clear()


    log_file_path = r"C:\Users\didri\Desktop\Programmering\VideoEnchancer program\local_model\Model_general_extra\log_inference_loading\Model_loading_inference_general.txt"

    file_handler = logging.FileHandler(log_file_path, mode='a')
    formatter = logging.Formatter('%(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger

logger = setup_logger()

def load_local_model(model_id):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    start_time = time.time()

    model = TransformersModel(
        model_id=model_id,
        device_map="cuda",
        torch_dtype=torch.float16 if not load_in_8bit else None,
        max_new_tokens=500,
        load_in_8bit=load_in_8bit,
    )

    load_duration = time.time() - start_time

    total_mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    mem_allocated_gb = torch.cuda.memory_allocated() / (1024 ** 3)
    mem_reserved_gb = torch.cuda.memory_reserved() / (1024 ** 3)
    mem_used_gb_after_load = torch.cuda.max_memory_allocated() / (1024 ** 3)

    logger.info(f"Model Loaded Successfully! \n Loading Time: {load_duration:.2f} seconds")
    logger.info(f"Total GPU Memory Available: {total_mem_gb:.2f} GB")
    logger.info(f"GPU Memory allocated (used): {mem_allocated_gb:.2f} GB")
    logger.info(f"GPU Memory reserved (cached): {mem_reserved_gb:.2f} GB")
    logger.info(f"GPU Memory used after Model Load: {mem_used_gb_after_load:.2f} GB")

    return model, load_duration, mem_used_gb_after_load

def run_model_prompt(Model, user_task):
    torch.cuda.reset_peak_memory_stats()
    Agent = CodeAgent(
        model=Model,
        verbosity_level=1,
        tools=[FinalAnswerTool(), DuckDuckGoSearchTool()]
    )
    start_time = time.time()
    result = Agent.run(user_task)
    inference_duration = time.time() - start_time

    mem_allocated_gb = torch.cuda.memory_allocated() / (1024 ** 3)
    mem_reserved_gb = torch.cuda.memory_reserved() / (1024 ** 3)
    mem_used_gb_after_inference = torch.cuda.max_memory_allocated() / (1024 ** 3)

    logger.info(f"Inference Time: {inference_duration:.2f} seconds.")
    logger.info(f"GPU Memory allocated (used): {mem_allocated_gb:.2f} GB")
    logger.info(f"GPU Memory reserved (cached): {mem_reserved_gb:.2f} GB")
    logger.info(f"GPU Memory used after Inference: {mem_used_gb_after_inference:.2f} GB")
    logger.info(f"user task: {user_task}")
    logger.info(f"Agent result:{result}")

    return result, inference_duration, mem_used_gb_after_inference

def test_inference_and_model_loading_time():
    model_path = r"C:\Users\didri\Desktop\Programmering\VideoEnchancer program\local_model\Qwen\Qwen2.5-Coder-7B-Instruct"
    clean_model_name = "Qwen2.5-Coder-7B-Instruct"
    global logger
    logger = setup_logger(clean_model_name)
    logger.info(f"Testing model: {clean_model_name}")
    
    user_task1 = "Hello What is your name?"
    
    model, load_time, mem_after_load = load_local_model(model_path)
    result1, inference_time1, mem_after_inference = run_model_prompt(model, user_task1)
    del model
    torch.cuda.empty_cache()
    gc.collect()

    return {
        "load_time": load_time,
        "mem_after_load": mem_after_load,
        "mem_after_inference": mem_after_inference,
        "task1": {
            "prompt": user_task1,
            "inference_time": inference_time1,
            "result": result1
        }
    }

if __name__ == "__main__":
    torch.cuda.empty_cache()
    gc.collect()
    # result = test_inference_and_model_loading_time()
    # total_time = result["load_time"] + result["task1"]["inference_time"]

    # logger.info("\nSummary:")
    # logger.info(f"Model load time: {result['load_time']:.2f}s")
    # logger.info(f"task 1 inference time: {result['task1']['inference_time']:.2f}s")
    # logger.info(f"Model used {result['mem_after_load']:.2f} GB of GPU memory to load.")
    # logger.info(f"Model used {result['mem_after_inference']:.2f} GB of GPU memory to do inference.")
    # logger.info(f"The total time it took is {total_time:.2f} seconds.")
    # logger.info(f"Model is loaded in {'8-bit quantization' if load_in_8bit else 'FP16 (half-precision)'}.")
    # logger.info("------------------------------------------------------------------------------------------------------------------------------------\n\n")
    # torch.cuda.empty_cache()
    # gc.collect()
    download_microsoft_microsoft_Phi_4_mini_instruct_onnx()


