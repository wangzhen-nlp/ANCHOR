from alarm_types import CRITICAL_ALARMS


DOMAIN_RAN = "RAN"
DOMAIN_TRANSMISSION = "TRANSMISSION"
DOMAIN_DATA = "DATA"


# Heuristic alarm-to-domain mapping for alarms defined in alarm_types.py.
# Ambiguous or management-side alarms are temporarily mapped to DATA so they
# can be reviewed and adjusted later.
ALARM_TO_DOMAIN = {
    # OFFLINE_ALARMS
    "BASE STATION FAULTY": DOMAIN_RAN,
    "BCF FAULTY": DOMAIN_RAN,
    "BN EMS Alarm NE Communication Failure": DOMAIN_DATA,
    "BTS Down": DOMAIN_RAN,
    "BTS O&M LINK FAILURE": DOMAIN_RAN,
    "Communication FAIL": DOMAIN_DATA,
    "Ericsson 2G NE Down": DOMAIN_RAN,
    "Ericsson 4G NE Down": DOMAIN_RAN,
    "Ericsson 4G S1 NE Down": DOMAIN_RAN,
    "Heartbeat Failure": DOMAIN_DATA,
    "Huawei 2G NE Down": DOMAIN_RAN,
    "Huawei 4G NE Down": DOMAIN_RAN,
    "Huawei 4G S1 NE Down": DOMAIN_RAN,
    "Loss of communications with NE": DOMAIN_DATA,
    "NE Is Disconnected": DOMAIN_DATA,
    "NE is Disconnected": DOMAIN_DATA,
    "NE O&M CONNECTION FAILURE": DOMAIN_DATA,
    "NE OM CONNECTION FAILURE": DOMAIN_DATA,
    "NE3SWS AGENT NOT RESPONDING TO REQUESTS": DOMAIN_DATA,
    "NE_COMMU_BREAK": DOMAIN_DATA,
    "NE_NOT_LOGIN": DOMAIN_DATA,
    "NodeB Down": DOMAIN_RAN,
    "Nokia 2G NE Down": DOMAIN_RAN,
    "Nokia 4G NE Down": DOMAIN_RAN,
    "Nokia 4G S1 NE Down": DOMAIN_RAN,
    "PMS Communication Failure": DOMAIN_DATA,
    "ReachabilityProblem": DOMAIN_DATA,
    "SWT_SWITCH_DOWN": DOMAIN_DATA,
    "The Device is offline": DOMAIN_DATA,
    "The link between the server and the NE is broken": DOMAIN_DATA,
    "WCDMA BASE STATION OUT OF USE": DOMAIN_RAN,
    "eNodeB Out of Service": DOMAIN_RAN,
    "gNodeB Out of Service": DOMAIN_RAN,

    # POWER_ALARMS
    "Device Powered Off": DOMAIN_DATA,
    "Ethernet Physical (ETPI) Remote dying gasp event": DOMAIN_TRANSMISSION,
    "Low Input Voltage": DOMAIN_TRANSMISSION,
    "Mains Failure": DOMAIN_RAN,
    "POWER_ABNORMAL": DOMAIN_DATA,
    "POWER_ALM Power Invalid": DOMAIN_DATA,
    "Power Fail(Entity)": DOMAIN_DATA,
    "Power Supply": DOMAIN_DATA,
    "RPU powered off": DOMAIN_RAN,
    "The main power supply abnormal": DOMAIN_DATA,
    "The voltage of the power supply is abnormal.": DOMAIN_DATA,
    "cseShutDownNotify": DOMAIN_DATA,
    "DC Low Voltage": DOMAIN_TRANSMISSION,
    "Genset Stop": DOMAIN_RAN,

    # LINK_ALARMS
    "CSL Fault": DOMAIN_TRANSMISSION,
    "E1 LOS": DOMAIN_TRANSMISSION,
    "ETH LOS": DOMAIN_TRANSMISSION,
    "ETH_LINK_DOWN": DOMAIN_TRANSMISSION,
    "ETH_LOS": DOMAIN_TRANSMISSION,
    "Ethernet Physical (ETPI) Interface down": DOMAIN_TRANSMISSION,
    "Ethernet Physical (ETPI) LOS": DOMAIN_TRANSMISSION,
    "Ethernet Physical (ETPI) Port down": DOMAIN_TRANSMISSION,
    "Ethernet port disconnected": DOMAIN_TRANSMISSION,
    "IF_CABLE_OPEN": DOMAIN_TRANSMISSION,
    "InterfaceDown": DOMAIN_DATA,
    "LINK_DOWN": DOMAIN_TRANSMISSION,
    "LOF": DOMAIN_TRANSMISSION,
    "Link Down": DOMAIN_TRANSMISSION,
    "Link Failure": DOMAIN_TRANSMISSION,
    "MW_LOF": DOMAIN_TRANSMISSION,
    "Microwave link critical alarm.": DOMAIN_TRANSMISSION,
    "OML FAULT-RXOTRX": DOMAIN_RAN,
    "OML Fault": DOMAIN_RAN,
    "PLA port link down alarm": DOMAIN_TRANSMISSION,
    "Physical Port Down": DOMAIN_TRANSMISSION,
    "S1AP Link Down": DOMAIN_RAN,
    "Transmission unit link break": DOMAIN_TRANSMISSION,
}


RAN_ALARMS = {
    alarm_name
    for alarm_name, domain in ALARM_TO_DOMAIN.items()
    if domain == DOMAIN_RAN
}

TRANSMISSION_ALARMS = {
    alarm_name
    for alarm_name, domain in ALARM_TO_DOMAIN.items()
    if domain == DOMAIN_TRANSMISSION
}

DATA_ALARMS = {
    alarm_name
    for alarm_name, domain in ALARM_TO_DOMAIN.items()
    if domain == DOMAIN_DATA
}


UNMAPPED_CRITICAL_ALARMS = sorted(set(CRITICAL_ALARMS) - set(ALARM_TO_DOMAIN))
EXTRA_MAPPED_ALARMS = sorted(set(ALARM_TO_DOMAIN) - set(CRITICAL_ALARMS))
