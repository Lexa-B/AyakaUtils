from typing import List

from langchain.schema.runnable import RunnableLambda

from ayaka_utils.Classes.EmoTensorModels import EmoTensorFull_CTXD, EmoTensor4DSlice_CTXD, EmoTensor2DSlice_CTXD, EmoTensor1DSlice_CTXD
#from Utils.Defs.pprint import pprint

################################################################################
## Configuration

SupportedTensorFileVersions = [
    "v0.0.1-EmoTensor Sliced Contextualized"
]

################################################################################
## Functions

def GetEmoterIndex(Transient: EmoTensor4DSlice_CTXD, Emoter: str) -> int:
    """
    Get the index of the emoter in the transient
    """
    return [i for i, d in enumerate(Transient.emoters) if d.emoter_user == Emoter][0]

def GetUserPrefName(UserID: str, ConvoUsers: List[dict]) -> str:
    """
    Get the preferred name of a user from the conversation users list
    """
    return [d['preferred_name'] for d in ConvoUsers if d['id'] == UserID][0]

def BuildReadableEmoDesc(Emotion: EmoTensor1DSlice_CTXD, EmoScaleLabels: dict, RenderContext=False) -> str:
    """
    Build a readable emotion description that makes it easy for the LLM to read. Can output the context if RenderContext is True. Can have custom label assignments, but defaults to the default labels
    """
    if Emotion.intensity >= 0.3:
        EmotionDesc = "\t* "

        # Intensity check – using Scale and Labels_En from config
        intensity_scale = EmoScaleLabels["Intensity"]["Scale"]
        intensity_labels = EmoScaleLabels["Intensity"]["Labels_En"]
        intensity_label = intensity_labels[0]  # Default to lowest
        for i, threshold in enumerate(intensity_scale):
            if Emotion.intensity > threshold:
                intensity_label = intensity_labels[i]
        EmotionDesc += f"{intensity_label} "

        # Valence check – using Scale and Labels_En from config
        valence_scale = EmoScaleLabels["Valence"]["Scale"]
        valence_labels = EmoScaleLabels["Valence"]["Labels_En"]
        if Emotion.valence <= valence_scale[0][0]:
            EmotionDesc += f"and {valence_labels[0][0]} "
        elif Emotion.valence <= valence_scale[0][1]:
            EmotionDesc += f"{valence_labels[0][1]} "
        elif Emotion.valence >= valence_scale[1][1]:
            EmotionDesc += f"and {valence_labels[1][1]} "
        elif Emotion.valence >= valence_scale[1][0]:
            EmotionDesc += f"{valence_labels[1][0]} "

        EmotionDesc += f"{Emotion.emotion} "

        # Arousal check – using Scale and Labels_En from config
        arousal_scale = EmoScaleLabels["Arousal"]["Scale"]
        arousal_labels = EmoScaleLabels["Arousal"]["Labels_En"]
        arousal_label = arousal_labels[0]  # Default to lowest
        for i, threshold in enumerate(arousal_scale):
            if Emotion.arousal > threshold:
                arousal_label = arousal_labels[i]
        EmotionDesc += f"with {arousal_label.lower()} arousal. "

        if RenderContext:
            EmotionDesc += f"(The context for this specific emotion was \"{Emotion.context}\")\n"
        else:
            EmotionDesc += "\n"
        return EmotionDesc
    
def BuildReadableEmoStateDesc(Emotions: EmoTensor2DSlice_CTXD, EmoScaleLabels: dict, RenderContext=False) -> str:
    """
    Build a readable emotion state description. Can output the context if RenderContext is True. Can have custom label assignments, but defaults to the default labels
    """
    Output = ""
    for Emotion in Emotions:
        this = BuildReadableEmoDesc(Emotion, RenderContext=RenderContext, EmoScaleLabels=EmoScaleLabels)
        if this:
            Output += this
    return Output

################################################################################
## History String Generators

def GenHistMessage(transient: EmoTensor4DSlice_CTXD, ConvoUsers: List[dict]) -> str:
    """
    Generate the 【USER】: MESSAGE string for the given transient
    """
    return f"【{GetUserPrefName(transient.speaker_user, ConvoUsers)}】：{transient.message}\n"

def GenHistEmoCtx(transient: EmoTensor4DSlice_CTXD, ConvoUsers: List[dict], PerspectiveUser: str) -> str:
    """
    Generate the emotional context for the given transient
    """
    Output = "* Emotional context:「\n"
    for targ in transient.emoters[GetEmoterIndex(transient, PerspectiveUser)].targets:
        try:
            ThisTarget_Name = GetUserPrefName(targ.this_target.split("-")[1], ConvoUsers)
        except Exception as e:
            raise ValueError(f"Could not find user in targ.this_target: {targ.this_target}") from e

        Output += f"* {GetUserPrefName(PerspectiveUser, ConvoUsers)}'s emotional context towards {ThisTarget_Name}:「{targ.scratch_context}」\n"
    return Output + "」\n"

def GenHistEmoSynopsis(transient: EmoTensor4DSlice_CTXD, ConvoUsers: List[dict], PerspectiveUser: str) -> str:
    """
    Generate the emotional synopsis for the given transient
    """
    Output = ""
    for targ in transient.emoters[GetEmoterIndex(transient, PerspectiveUser)].targets:
        try:
            ThisTarget_Name = GetUserPrefName(targ.this_target.split("-")[1], ConvoUsers)
        except Exception as e:
            raise ValueError(f"Could not find user in targ.this_target: {targ.this_target}") from e
        Output += f"* {GetUserPrefName(PerspectiveUser, ConvoUsers)}'s emotional synopsis towards {ThisTarget_Name}:「{targ.scratch_synopsis}」\n"
    return Output + "」\n"

def GenHistExtCtx_PerspUser(transient: EmoTensor4DSlice_CTXD, ConvoUsers: List[dict], PerspectiveUser: str) -> str:
    """
    Generate the external context for the perspective user in the given transient
    """
    Output = ""
    for emoter in transient.emoters:
        if emoter.emoter_user == PerspectiveUser:
            Output += f"* {GetUserPrefName(emoter.emoter_user, ConvoUsers)}'s external context:「{emoter.external_context}」\n"
    return Output

def GenHistExtCtx_OtherUsers(transient: EmoTensor4DSlice_CTXD, ConvoUsers: List[dict], PerspectiveUser: str) -> str:
    """
    Generate the external context for all other users in the given transient
    """
    Output = ""
    for emoter in transient.emoters:
        if emoter.emoter_user != PerspectiveUser:
            Output += f"* {GetUserPrefName(emoter.emoter_user, ConvoUsers)}'s external context:「{emoter.external_context}」\n"
    return Output

def GenHistEmoStateDesc_PerspUser(transient: EmoTensor4DSlice_CTXD, ConvoUsers: List[dict], PerspectiveUser: str, RenderContext: bool) -> str:
    """
    Generate the emotional state description for the perspective user in the given transient
    """
    Output = ""
    for emoter in transient.emoters:
        if emoter.emoter_user == PerspectiveUser:
            Output += f"* {GetUserPrefName(emoter.emoter_user, ConvoUsers)}'s emotions:\n{BuildReadableEmoStateDesc(emoter.targets[0].emotions, RenderContext=RenderContext)}\n"
    return Output

def GenHistEmoStateDesc_OtherUsers(transient: EmoTensor4DSlice_CTXD, ConvoUsers: List[dict], PerspectiveUser: str, RenderContext: bool) -> str:
    """
    Generate the emotional state description for all other users in the given transient
    """
    Output = ""
    for emoter in transient.emoters:
        if emoter.emoter_user != PerspectiveUser:
            Output += f"* {GetUserPrefName(emoter.emoter_user, ConvoUsers)}'s emotions:\n{BuildReadableEmoStateDesc(emoter.targets[0].emotions, RenderContext=RenderContext)}\n"
    return Output

################################################################################
## Get a simplified conversation history for the LLM to read

def GetSimplifiedConversationHistory(
        TensorFile: EmoTensorFull_CTXD, 
        ConvoUsers: List[dict], 
        PerspectiveUser: str,
        EmoScaleLabels: dict,
        Fidelity_1: int = 2, 
        Fidelity_2: int = 2, 
        Fidelity_3: int = 4, 
        Fidelity_4: int = 8, 
        Fidelity_5: int = 16,
        Fidelity_6: int = 32) -> str:
    """
    Get a simplified conversation history for the LLM to read
    """

    if isinstance(TensorFile, dict):  # If it's a dict, convert it to the Pydantic model
        TensorFile = EmoTensorFull_CTXD.model_validate(TensorFile)

    try:
        TF_Ver = TensorFile.version
        # TF_O1_Attr = TensorFile.order_1_attributes
        # TF_O2_Emos = TensorFile.order_2_emotions
        # TF_O3_Targ = TensorFile.order_3_target
        # TF_O4_Emots = TensorFile.order_4_emoters
        # TF_O5_Trns = TensorFile.order_5_transients
        # TF_UserPrefNames = TensorFile.user_prefnames
        TF_Transients = TensorFile.transients
    except AttributeError:
        raise ValueError(f"TensorFile must be either a dict that can be converted to EmoTensorFull_CTXD or have version/transients attributes. Got: {type(TensorFile)}")

    if TF_Ver not in SupportedTensorFileVersions:
        raise ValueError(f"Unsupported TensorFile version: {TF_Ver}")

    output = ""
    for i, transient in enumerate(reversed(TF_Transients)):

        if i < (Fidelity_1): # Full-fidelity message. Show a lot of detail, including the long-form emotional context.
            output = (
                f"Message {len(TF_Transients) - i}:\n" + 
                GenHistMessage(transient, ConvoUsers) +
                GenHistEmoCtx(transient, ConvoUsers, PerspectiveUser) +
                GenHistExtCtx_PerspUser(transient, ConvoUsers, PerspectiveUser) +
                GenHistExtCtx_OtherUsers(transient, ConvoUsers, PerspectiveUser) +
                GenHistEmoStateDesc_PerspUser(transient, ConvoUsers, PerspectiveUser, RenderContext=True) +
                GenHistEmoStateDesc_OtherUsers(transient, ConvoUsers, PerspectiveUser, RenderContext=True)
            ) + output # Prepend to output

        elif i < (Fidelity_1 + Fidelity_2): # High-fidelity message. Show a lot of detail, including the summarized emotional context.
            output = (
                f"Message {len(TF_Transients) - i}:\n" + 
                GenHistMessage(transient, ConvoUsers) +
                GenHistEmoSynopsis(transient, ConvoUsers, PerspectiveUser) +
                GenHistExtCtx_PerspUser(transient, ConvoUsers, PerspectiveUser) +
                GenHistExtCtx_OtherUsers(transient, ConvoUsers, PerspectiveUser) +
                GenHistEmoStateDesc_PerspUser(transient, ConvoUsers, PerspectiveUser, RenderContext=True) +
                GenHistEmoStateDesc_OtherUsers(transient, ConvoUsers, PerspectiveUser, RenderContext=False)
            ) + output # Prepend to output
        
        elif i < (Fidelity_1 + Fidelity_2 + Fidelity_3): # Medium-fidelity message. Show some detail, including the summarized emotional context.
            output = (
                f"Message {len(TF_Transients) - i}:\n" + 
                GenHistMessage(transient, ConvoUsers) +
                GenHistEmoSynopsis(transient, ConvoUsers, PerspectiveUser) +
                GenHistExtCtx_PerspUser(transient, ConvoUsers, PerspectiveUser) +
                GenHistExtCtx_OtherUsers(transient, ConvoUsers, PerspectiveUser) +
                GenHistEmoStateDesc_PerspUser(transient, ConvoUsers, PerspectiveUser, RenderContext=True) +
                GenHistEmoStateDesc_OtherUsers(transient, ConvoUsers, PerspectiveUser, RenderContext=False)
            ) + output # Prepend to output
        
        elif i < (Fidelity_1 + Fidelity_2 + Fidelity_3 + Fidelity_4): # Low-fidelity message. Show a bit of detail, including the summarized emotional context.
            output = (
                f"Message {len(TF_Transients) - i}:\n" + 
                GenHistMessage(transient, ConvoUsers) +
                GenHistEmoSynopsis(transient, ConvoUsers, PerspectiveUser) +
                GenHistExtCtx_PerspUser(transient, ConvoUsers, PerspectiveUser) +
                GenHistEmoStateDesc_PerspUser(transient, ConvoUsers, PerspectiveUser, RenderContext=False)
            ) + output # Prepend to output
        
        elif i < (Fidelity_1 + Fidelity_2 + Fidelity_3 + Fidelity_4 + Fidelity_5): # Lower-fidelity message. Show a tiny bit of detail.
            output = (
                f"Message {len(TF_Transients) - i}:\n" + 
                GenHistMessage(transient, ConvoUsers) +
                GenHistEmoStateDesc_PerspUser(transient, ConvoUsers, PerspectiveUser, RenderContext=False)
            ) + output # Prepend to output

        elif i < (Fidelity_1 + Fidelity_2 + Fidelity_3 + Fidelity_4 + Fidelity_5 + Fidelity_6): # Lowest-fidelity message. Only show the message.
            output = (
                f"Message {len(TF_Transients) - i}:\n" + 
                GenHistMessage(transient, ConvoUsers)
            ) + output # Prepend to output
        output = "\n" + output

    return output


def REmoConvHist(
            TensorFile: EmoTensorFull_CTXD, 
            ConvoUsers: List[dict],
            EmoScaleLabels: dict, 
            Fidelity_1: int = 2, 
            Fidelity_2: int = 2, 
            Fidelity_3: int = 4, 
            Fidelity_4: int = 8, 
            Fidelity_5: int = 16,
            Fidelity_6: int = 32) -> RunnableLambda:
    """
    Return a runnable that returns a simplified conversation history for the LLM to read
    """

    return RunnableLambda(GetSimplifiedConversationHistory(
        TensorFile=TensorFile,
        ConvoUsers=ConvoUsers,
        Fidelity_1=Fidelity_1,
        Fidelity_2=Fidelity_2,
        Fidelity_3=Fidelity_3,
        Fidelity_4=Fidelity_4,
        Fidelity_5=Fidelity_5,
        EmoScaleLabels=EmoScaleLabels
    ))
