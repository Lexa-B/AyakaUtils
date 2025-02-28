import os
import json
from functools import partial

# LangChain
from langchain.schema.runnable import RunnableLambda
from langchain.schema.output_parser import StrOutputParser

# Other
import tiktoken

# Custom
from ayaka_utils.Defs.pprint import pprint

################################################################################
## Constants

ModelName = "gpt-4o-mini"

################################################################################
## Defs

def GetTokenCount(string: str) -> int:
    """Get the number of tokens in a string using a specified encoding"""
    encoding = tiktoken.encoding_for_model(ModelName)
    num_tokens = len(encoding.encode(string))
    return num_tokens

################################################################################
## Runnable

def RSave(RunningState, filename, source=None, preface="", path="./", filetype="json", overwrite=False, suppress_save=False, suppress_print=False, verbose=False):
    def SaveToFile(RunningState, filename, source, preface, path, filetype, overwrite, suppress_save, suppress_print, verbose):
        if not RunningState:
            raise ValueError("RunningState is required")
        if not filename:
            raise ValueError("Filename is required")
        
        if source is not None:
            if hasattr(RunningState, "__getitem__"):
                input_text = RunningState[source]
            else:
                pprint(f"ERROR: RunningState is not subscriptable. Could not pull requested source \"{source}\" from RunningState, so saving RunningState as a whole.")
                input_text = RunningState
        else:
            input_text = RunningState

        def get_text(x):
            if hasattr(x, "messages"):
                return "\n".join(m.content for m in x.messages)
            else:
                return str(x)

        raw_text = get_text(input_text)
        parsed_text = StrOutputParser().parse(raw_text)

        ## Done loading the text, now acting on it

        if source is None: # If source is not specified, use "RunningState"
            source = "RunningState"

        if not suppress_print:
            PrintText = (f"{preface}:\nIf model is {ModelName}, then the token count is {GetTokenCount(parsed_text)}\n")
            if not suppress_save:
                PrintText += (f"Saving {source} to '{filename}' in '{path}'\nIt ")
            else:
                PrintText += (f"Not saving {source} due to suppress_save attribute, but it ")
            PrintText += (f"contains {len(parsed_text)} characters")
            if verbose:
                PrintText += (f":\n\n\"\"\"\n{parsed_text}\n\"\"\"")
            pprint(PrintText)

        if not suppress_save:
            # Build the file path using os.path.join for OS-independent paths
            if filetype == "json":
                file_path = os.path.join(path, filename + ".json")
            elif filetype == "txt":
                file_path = os.path.join(path, filename + ".txt")
            else:
                raise ValueError("Unsupported file type")
            
            # Create the directory if it doesn't exist
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            # Write to the file
            if filetype == "json":
                with open(file_path, "w", encoding="utf-8") as f:
                    output_dict = {
                        "source": source,
                        "preface": preface,
                        "header": PrintText,
                        "text": parsed_text
                    }
                    json.dump(output_dict, f)
            elif filetype == "txt":
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(PrintText + "\n\n" + "="*100 + "\n\n" + parsed_text)

        return RunningState

    return RunnableLambda(partial(SaveToFile, filename=filename, source=source, preface=preface, path=path, filetype=filetype, overwrite=overwrite, suppress_save=suppress_save, suppress_print=suppress_print, verbose=verbose))
