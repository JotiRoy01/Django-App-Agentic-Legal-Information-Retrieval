from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
import torch
from agentic.data_loader.data_loader import load
from agentic.exception import Agentic_Exception
import sys

class QueryExpansion :
    def __init__(self) :
        try :
            # load the dataset
            self.val_df = load(filename="val.csv")
        except Exception as e :
            raise Agentic_Exception(e, sys) from e

    def Eng_plus_Germ(self) :
        # Select a model appropriate for the current hardware.
        if torch.cuda.is_available():
            model_id = "Qwen/Qwen2.5-1.5B-Instruct"
            load_kwargs = {
                "torch_dtype": torch.float16,
                "device_map": "auto",
            }
        else:
            model_id = "distilgpt2"
            load_kwargs = {
                "torch_dtype": torch.float32,
                "device_map": "cpu",
            }

        try :
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            if model_id == "google/flan-t5-small":
                model = AutoModelForSeq2SeqLM.from_pretrained(
                    model_id,
                    **load_kwargs,
                )
                agent_pipe = pipeline(
                    "text2text-generation",
                    model=model,
                    tokenizer=tokenizer,
                    max_new_tokens=100,
                    do_sample=False,
                )
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    **load_kwargs,
                )
                agent_pipe = pipeline(
                    "text-generation",
                    model=model,
                    tokenizer=tokenizer,
                    max_new_tokens=100,
                    do_sample=False,
                )
        except Exception as e :
            # Fallback to a smaller CPU-safe causal model when the preferred model cannot be loaded.
            fallback_model = "distilgpt2"
            try:
                tokenizer = AutoTokenizer.from_pretrained(fallback_model)
                model = AutoModelForCausalLM.from_pretrained(
                    fallback_model,
                    torch_dtype=torch.float32,
                    device_map="cpu",
                )
                agent_pipe = pipeline(
                    "text-generation",
                    model=model,
                    tokenizer=tokenizer,
                    max_new_tokens=100,
                    do_sample=False,
                )
                model_id = fallback_model
            except Exception as fallback_e:
                raise Agentic_Exception(
                    f"Failed to load expansion model: {e} | fallback error: {fallback_e}",
                    sys,
                ) from fallback_e

        test_query = self.val_df['query'].iloc[0]
        # Sanitize Unicode for Windows console
        safe_query = str(test_query)[:150].replace('\u2011', '-').replace('\u2012', '-').replace('\u2013', '-').replace('\u2014', '-')
        print(f"Original Query: {safe_query}...")

        prompt = (
            "You are an expert Swiss lawyer. Extract the core legal concepts from the English query "
            "and translate them into German keywords for a database search. Also list any relevant Swiss law "
            "abbreviations (like StPO, ZGB, OR, StGB).\n\n"
            f"Query: {test_query}\n\nProvide the German keywords and law abbreviations only:"
        )

        outputs = agent_pipe(prompt)
        generated_text = outputs[0].get("generated_text", "").strip()

        print(f"Using model {model_id} for expansion")
        # Sanitize generated text for logging
        safe_gen = generated_text[:150].replace('\u2011', '-').replace('\u2012', '-').replace('\u2013', '-').replace('\u2014', '-')
        print(f"Generated text: {safe_gen}...")
        return generated_text