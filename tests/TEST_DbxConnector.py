import sys
import os
import json
from ayaka_utils.Defs.pprint import pprint

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ayaka_utils.Defs.DbxConnector import DbxFIO

DataToSend = json.load(open('./tests/EmoTensorDB/EmoTensor-TEST.etsc', 'r'))

DbxFIO('/EmoTensor/data/Sliced-Contextualized/EmoTensor-TEST.etsc', mode="write", type="json", data=DataToSend)
pprint(DbxFIO('/EmoTensor/data/Sliced-Contextualized/EmoTensor-TEST.etsc', mode="read", type="json"))
