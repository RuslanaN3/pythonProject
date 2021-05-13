from api.pred.cnn_functions import load_patches_coord


def convert_data(predictions):
    patches_coord = load_patches_coord("api/resources/camera5s.csv")
    parking_slots_states = []
    for indx, pc in enumerate(patches_coord):
        slot_id = pc["SlotId"]
        slot_state = "VACANT" if predictions[indx] == 0 else "OCCUPIED"
        parking_slots_states.append({"slotNumber": slot_id, "slotStatus": slot_state})
    return parking_slots_states

# {
#         "slotNumber": 3918,
#         "slotState": "VACANT",
# }
