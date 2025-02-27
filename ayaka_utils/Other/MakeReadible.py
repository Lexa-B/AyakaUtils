import ast
import json

# inputs = ["./Utils/Other/MakeReadible/CurrentPerceiverEmoContext.txt",
# "./Utils/Other/MakeReadible/CurrentPerceiverEmotion.txt",
# "./Utils/Other/MakeReadible/CurrentPerceiverScratchpad.txt",
# "./Utils/Other/MakeReadible/CurrentSpeakerEmoContext.txt",
# "./Utils/Other/MakeReadible/CurrentSpeakerEmotion.txt",
# "./Utils/Other/MakeReadible/CurrentSpeakerScratchpad.txt"]

# outputs = ["./Utils/Other/MakeReadible/formatted_CurrentPerceiverEmoContext.txt",
# "./Utils/Other/MakeReadible/formatted_CurrentPerceiverEmotion.txt",
# "./Utils/Other/MakeReadible/formatted_CurrentPerceiverScratchpad.txt",
# "./Utils/Other/MakeReadible/formatted_CurrentSpeakerEmoContext.txt",
# "./Utils/Other/MakeReadible/formatted_CurrentSpeakerEmotion.txt",
# "./Utils/Other/MakeReadible/formatted_CurrentSpeakerScratchpad.txt"]

inputs = ["./Utils/Other/MakeReadible/01.CurrentScratchpad.txt"]

outputs = ["./Utils/Other/MakeReadible/formatted_CurrentScratchpad.txt"]
outputs_json = ["./Utils/Other/MakeReadible/formatted_CurrentScratchpad.json"]


for i in range(len(inputs)):
    path = inputs[i]
    output_path = outputs[i]
    output_json_path = outputs_json[i]
    with open(path, "r", encoding="utf-8") as f:
        raw_text = f.read()

    # Use repr() to build a literal string so that ast.literal_eval works safely
    literal_string = "{'data': " + repr(raw_text) + "}"
    d = ast.literal_eval(literal_string)

    def custom_unescape(s):
        # Replace common escape sequences with their actual characters.
        # Order matters here; we start with those that don’t risk interfering with others.
        s = s.replace("\\'", "'")   # Convert \' to '
        s = s.replace('\\"', '"')   # Convert \" to "
        s = s.replace("\\n", "\n")  # Convert \n to an actual newline
        s = s.replace("\\t", "\t")  # Convert \t to a tab
        s = s.replace("\\r", "\r")  # Convert \r to carriage return
        return s

    unescaped_content = custom_unescape(d["data"])

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(unescaped_content)

    with open(output_json_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(d, indent=2, ensure_ascii=False))
