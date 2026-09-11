import argparse
import json
import sys

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    args = parser.parse_args()

    false_sat = 0
    with open(args.input, "r") as f:
        for line in f:
            if not line.strip(): continue
            record = json.loads(line)
            # A false sat is when a solver reports SAT but the verifier found violations > 0
            if record["status"] == "SAT_VERIFIED" and record.get("violations", -1) > 0:
                print(f"FALSE SAT DETECTED: {record['solver']} on {record['instance']}")
                false_sat += 1

    print(f"false_sat_count = {false_sat}")
    if false_sat > 0:
        sys.exit(1)

if __name__ == "__main__":
    main()
