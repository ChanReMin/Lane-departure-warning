import lgpio
import time

h = lgpio.gpiochip_open(0)
LED_GPIO = 3

# Claim output only once
lgpio.gpio_claim_output(h, LED_GPIO)

try:
    while True:
        # Turn ON
        lgpio.gpio_write(h, LED_GPIO, 1)
        time.sleep(0.1)

        # Turn OFF
        lgpio.gpio_write(h, LED_GPIO, 0)
        time.sleep(0.1)

except KeyboardInterrupt:
    print("Exiting...")

finally:
    lgpio.gpio_write(h, LED_GPIO, 0)
    lgpio.gpiochip_close(h)
