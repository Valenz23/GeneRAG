from typing import Text, List, Any, Dict

from rasa_sdk import Tracker, FormValidationAction
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.types import DomainDict

import re

class ValidateDatosDNIForm(FormValidationAction):
    def name(self) -> Text:
        return "validate_datosDNI_form"
    
    def validate_nombre(self,
                        slot_value: Any,
                        dispatcher: CollectingDispatcher,
                        tracker: Tracker,
                        domain: DomainDict) -> Dict[Text, Any]:
        
        if slot_value: 
            return {"nombre":slot_value}
        else: 
            return {"nombre":None}
        
    def validate_dni(self,
                        slot_value: Any,
                        dispatcher: CollectingDispatcher,
                        tracker: Tracker,
                        domain: DomainDict) -> Dict[Text, Any]:
        
        regex = "^[0-9]{8}[A-Z]{0,1}$"
        
        if slot_value:
            if re.search(regex, slot_value):     
                return {"dni":slot_value}
            else:
                dispatcher.utter_message(template="utter_dniNoValido")            
                return {"dni":None}