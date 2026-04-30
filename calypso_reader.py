import serial
import time




DIRECTION_OFFSET_DEGREES = 287

def apply_direction_offset(raw_direction: float) -> float:
    return (raw_direction + DIRECTION_OFFSET_DEGREES) % 360


def nmea_checksum_ok(sentence: str) -> bool:
    sentence = sentence.strip()
    if not sentence.startswith("$") or "*" not in sentence:
        return False

    body, checksum = sentence[1:].split("*", 1)

    calc = 0
    for ch in body:
        calc ^= ord(ch)

    try:
        return calc == int(checksum[:2], 16)
    except ValueError:
        return False


def parse_mwv(sentence: str):
    sentence = sentence.strip()

    if not nmea_checksum_ok(sentence):
        return None

    body = sentence[1:sentence.index("*")]
    parts = body.split(",")

    if len(parts) != 6:
        return None

    sentence_type = parts[0]
    if not sentence_type.endswith("MWV"):
        return None

    try:
        direction_deg = float(parts[1])
        reference = parts[2]
        speed = float(parts[3])
        units = parts[4]
        status = parts[5]
    except ValueError:
        return None

    if status != "A":
        return None

    return {
        "direction_deg": direction_deg,
        "reference": reference,
        "speed": speed,
        "units": units,
        "status": status,
    }


def to_knots(speed: float, units: str) -> float | None:
    units = units.upper()

    if units == "N":
        return speed
    if units == "M":
        return speed * 1.94384
    if units == "K":
        return speed * 0.539957

    return None


class CalypsoReader:
    def __init__(self, port="/dev/serial0", baudrate=38400, timeout=1):
        self.ser = serial.Serial(port, baudrate, timeout=timeout)

    def read_once(self):
        raw = self.ser.readline().decode("ascii", errors="ignore").strip()
        if not raw:
            return None

        parsed = parse_mwv(raw)
        if not parsed:
            return None


        speed_knots = to_knots(parsed["speed"], parsed["units"])
        if speed_knots is None:
            return None

        corrected_direction = apply_direction_offset(parsed["direction_deg"])

        return {
            "timestamp": time.time(),
            "wind_speed_kt": round(speed_knots, 2),
            "wind_direction_deg": round(corrected_direction, 1),
            "raw_speed": parsed["speed"],
            "raw_units": parsed["units"],
            "reference": parsed["reference"],
            "raw_sentence": raw,
        }

