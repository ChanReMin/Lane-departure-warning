import lgpio
import time

h = lgpio.gpiochip_open(0)
BUZZER_GPIO = 17

# Claim output only once
lgpio.gpio_claim_output(h, BUZZER_GPIO)

try:
    while True:
        # Turn ON
        lgpio.gpio_write(h, BUZZER_GPIO, 1)
        time.sleep(1)

        # Turn OFF
        lgpio.gpio_write(h, BUZZER_GPIO, 0)
        time.sleep(1)

except KeyboardInterrupt:
    print("Exiting...")

finally:
    lgpio.gpio_write(h, BUZZER_GPIO, 0)
    lgpio.gpiochip_close(h)
