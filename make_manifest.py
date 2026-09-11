import os
import glob

data_dir = "data/normalized"
out_file = "data/frozen/instances_clean.csv"

with open(out_file, "w") as f:
    f.write("instance,family,status,relaxed,reason\n")
    for filepath in glob.glob(os.path.join(data_dir, "*.json")):
        fname = os.path.basename(filepath)
        family = fname.split("-")[0] if "-" in fname else fname.split("_")[0]
        f.write(f"{fname},{family},clean,False,\n")
print(f"Wrote {out_file}")
