from typing import Any, Text, Dict, List

from rasa_sdk import Tracker, FormValidationAction, Action, ValidationAction
from rasa_sdk.events import EventType, SlotSet, AllSlotsReset
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.types import DomainDict

import dateparser
from datetime import datetime

DB_NAME = {
    'Ngô Mạnh Tiến' : 'TS.',
    'Hà Kim Duyên' : 'TS.',
    'Vũ Minh Đức' : 'TS.'
}

ALLOWED_SLOTS = ['tên', 'thời gian']

def time_extract(sentence):
    time = dateparser.parse(sentence, languages = ['vi'], region = 'vi', locales = ['vi'])
    return time

class ValidateDatLichForm(FormValidationAction):

    def name(self) -> Text:
        return 'validate_dat_lich_form'

    def validate_name(
        self,
        slot_value: Any,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: DomainDict,
    ) -> Dict[Text, Any]:
        first_names = {first_name.split(' ')[-1].lower() : first_name for first_name in list(DB_NAME.keys())}
        if slot_value.split(' ')[-1].lower() not in list(first_names.keys()):
            dispatcher.utter_message(text = f'Tôi không tìm thấy tên người mà bạn muốn đặt lịch')
            return {'name': None}
        return {'name': f'{DB_NAME[first_names[slot_value.split(" ")[-1].lower()]]} {first_names[slot_value.split(" ")[-1].lower()]}'}
    
    def validate_time(
        self,
        slot_value: Any,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: DomainDict,
    ) -> Dict[Text, Any]:
        time = time_extract(slot_value)
        temp = True if time == None else False
        if temp or time <= datetime.now():
            dispatcher.utter_message(text = f'Thời gian đặt lịch của bạn không hợp lệ')
            return {'time': None}
        return {'time': f'{time.hour} giờ {time.minute} phút, ngày {time.day} tháng {time.month} năm {time.year}'}
    
    def validate_support(
        self,
        slot_value: Any,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: DomainDict,
    ) -> Dict[Text, Any]:
        if tracker.get_intent_of_latest_message() == 'dong_y':
            return {'support': True}
        if tracker.get_intent_of_latest_message() == 'tu_choi':
            dispatcher.utter_message(text = 'Vui lòng nhập lại thông tin')
            return {'name': None, 'time': None, 'support': None}
        dispatcher.utter_message(text = 'Bạn chỉ cần xác nhận đúng hay sai')
        return {'support': None}

class ResetSlots(Action):
    def name(self) -> Text:
        return 'action_reset_slot'

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict
    ) -> List[EventType]:
        if tracker.get_slot('name') != None and tracker.get_slot('time') != None and tracker.get_slot('support') != None:
            AllSlotsReset()
        return []

# class DatLich(Action):
#     def name(self) -> Text:
#         return 'action_dat_lich'

#     def run(
#         self,
#         dispatcher: CollectingDispatcher,
#         tracker: Tracker,
#         domain: Dict
#     ) -> List[EventType]:
#         pass
#         return []
