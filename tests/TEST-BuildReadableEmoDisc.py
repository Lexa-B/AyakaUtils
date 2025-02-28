from ayaka_utils.Classes.EmoTensorModels import EmoTensor1DSlice_CTXD
from ayaka_utils.Runnables.REmoConvHist import BuildReadableEmoDesc

print(BuildReadableEmoDesc(
    EmoTensor1DSlice_CTXD(
        emotion="喜び", 
        intensity=0.5, 
        valence=0.5, 
        arousal=0.5, 
        context="喜びの文脈"
    ),
    EmoScaleLabels={
        "Intensity": {
            "Definition": "Emotional intensity refers to the magnitude or strength with which an emotion is experienced, independent of its qualitative tone or activation level. For instance, within the anger family, one might feel a mild irritation as a low-intensity form versus a pronounced anger as a high-intensity form, both maintaining the same negative valence and arousal level. This variation reflects only the strength of the emotion without altering its underlying character.",
            "ScaleDesc": "Logarithmic scale, 0 is nonexistent, and asymptoting 1 is the highest possible intensity to the point that no human has ever reached.",
            "Scale": [0.00, 0.40, 0.50, 0.60, 0.80],
            "Labels_En": ["Slight", "Noticeable", "Moderate", "High", "Extreme"],
            "Labels_Ja": ["わずか", "目立つ", "中", "高", "非常"]
        },
        "Valence": {
            "Definition": "Emotional valence describes the inherent positivity or negativity of an emotion, independent of its strength or activation; for example, in the surprise family, a delightful surprise (positive valence) contrasts with an alarming surprise (negative valence) even when both are experienced with the same intensity and arousal.",
            "ScaleDesc": "Sigmoid scale, -1 is the most negative possible, 0 is neutral, and 1 is the most positive possible.",
            "Scale": [[-0.60, -0.30], [0.30, 0.60]],
            "Labels_En": [["Very Negative", "Negative"], ["Positive", "Very Positive"]],
            "Labels_Ja": [["非常負", "負"], ["正", "非常正"]]
        },
        "Arousal": {
            "Definition": "Arousal pertains to the level of physiological and psychological activation accompanying an emotion, ranging from calm to highly energized states. For example, within the joy family, a serene sense of contentment represents a low-arousal form of joy, while exuberance or elation represents a high-arousal form, both maintaining the same positive valence and intensity. This variation reflects only the degree of energetic activation without altering the emotion's inherent character.",
            "ScaleDesc": "Logistic scale, 0 is neutral, and 1 is the most positive possible. Values close to 1 are so high that the person is hysterical.",
            "Scale": [0.00, 0.40, 0.60, 0.80],
            "Labels_En": ["Low", "Moderate", "High", "Very High"],
            "Labels_Ja": ["低", "中", "高", "非常"]
        },
        "PromptGenLang": "En"
    }, 
    RenderContext=True, 
    LangCode="Ja"
))