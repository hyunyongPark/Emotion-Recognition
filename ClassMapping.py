import json

def JsonToDict():
    with open("Data/decoder_label.json", "r") as st_json:
        decoder_json = json.load(st_json)
    decoder_json.values()
    decoder_json
    return decoder_json

def Decoder(pred):
    mapping_dict = JsonToDict()
    return mapping_dict[str(pred)]