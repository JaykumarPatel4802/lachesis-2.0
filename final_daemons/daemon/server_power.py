import powertop
import json
import re
import time
import pandas as pd

def main():
    pattern = r"[-+]?\d*\.\d+|\d+"
    with open("power_data.csv", "w+") as f:
        # try:
        #     # powertop_obj = powertop.Powertop()
        #     for i in range(5):
        #         timestamp = time.time()
        #         print("start reading")
        #         try:
        #             # measures = powertop_obj.get_measures(time=0.05)
        #             measures = powertop.Powertop().get_measures(time=0.05)
        #             print("end reading")
        #             power = 0
        #             for entry in measures['Device Power Report']:
        #                 power_entry = entry['PW Estimate']
        #                 matches = re.findall(pattern, power_entry)
        #                 power += float(matches[0]) if matches else 0.0
        #             f.write(f'{timestamp}, {power}\n')
        #         except:
        #             f.write(f'{timestamp}, {-1}\n')
        # except:
        #     print("Finished getting readings")

        for i in range(200):
            timestamp = time.time()
            measures = powertop.Powertop().get_measures(time=1)
            power = 0
            for entry in measures['Device Power Report']:
                power_entry = entry['PW Estimate']
                matches = re.findall(pattern, power_entry)
                power += float(matches[0]) if matches else 0.0
            f.write(f'{timestamp}, {power}\n')

if __name__ == '__main__':
    main()