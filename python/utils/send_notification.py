from utils import send_mail
import argparse
from datetime import datetime

parser = argparse.ArgumentParser(description="This Program will send a mail when called with the time of execution "
                                             "and maybe a log file of ouputs and errors included")
# parser.add_argument("-v", "--verbose", help="set verbosity", action="count", default=0)
# parser.add_argument("-v", "--verbose", help="set verbosity", action="store_true")
parser.add_argument("-v", "--verbose", help="set verbosity", type=int, choices=[0, 1, 2])
# parser.add_argument("number", help="square a number that you input", type=int)
args = parser.parse_args()
# number = args.number
log_file = "epoch 1\nepoch2\nepoch3\n"
text = ""
print(args)
if args.verbose == 0:
    text = "program has finished"
elif args.verbose == 1:
    text = f"program has finished at {datetime.now().strftime('%m/%d/%Y, %H:%M h')}"
elif args.verbose >= 2:
    text = f"program has finished at {datetime.now().strftime('%m/%d/%Y, %H:%M h')}\n{log_file}"
print(text)

send_mail("michael.reiter94@gmail.com", text=text, subject="Slurm Notification")
