import lgpio
import time
# ---- Buzzer warning helper using lgpio ----
h = lgpio.gpiochip_open(0)
BUZZER_GPIO = 18
DETECTION_LED = 23
DEPARTURE_LED_R = 3
DEPARTURE_LED_L = 2

# Claim output only once
lgpio.gpio_claim_output(h, BUZZER_GPIO)
lgpio.gpio_claim_output(h, DETECTION_LED)
lgpio.gpio_claim_output(h, DEPARTURE_LED_L)
lgpio.gpio_claim_output(h, DEPARTURE_LED_R)

def detection_warning(level):
    if level not in (1, 2):
        return
    # Determine on/off durations
    if level == 2:
        on_time = 1
        off_time = 1
    else:
        on_time = 2
        off_time = 2
    # Simple beep via GPIO write
    lgpio.gpio_write(h, BUZZER_GPIO, 1)
    lgpio.gpio_write(h, DETECTION_LED, 1)
    time.sleep(on_time)
    lgpio.gpio_write(h, BUZZER_GPIO, 0)
    lgpio.gpio_write(h, DETECTION_LED, 0)
    time.sleep(off_time)

def departure_warning(lane):
    """
    Shared buzzer/LED warning for lane departure:
      - 'r': right lane departure
      - 'l': left lane departure
    Uses same buzzer pin with distinct timing
    """
    if lane == 'r':
        lgpio.gpio_write(h, BUZZER_GPIO, 1)
        lgpio.gpio_write(h, DEPARTURE_LED_R, 1)
        time.sleep(0.5)
        lgpio.gpio_write(h, BUZZER_GPIO, 0)
        lgpio.gpio_write(h, DEPARTURE_LED_R, 0)
        time.sleep(0.5)
    if lane == 'l':
        lgpio.gpio_write(h, BUZZER_GPIO, 1)
        lgpio.gpio_write(h, DEPARTURE_LED_L, 1)
        time.sleep(0.5)
        lgpio.gpio_write(h, BUZZER_GPIO, 0)
        lgpio.gpio_write(h, DEPARTURE_LED_L, 0)
        time.sleep(0.5)