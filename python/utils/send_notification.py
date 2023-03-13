from utils import send_mail
import argparse
from datetime import datetime

# <editor-fold desc="Argument Parser">
parser = argparse.ArgumentParser(description="This Program will send a mail when called with the time of execution "
                                             "and if specified alongside a log file of ouputs or errors")
parser.add_argument("-out", "--outputLogFile", help="adress to an output log text-file to append to the text body of "
                                                    "the email")
parser.add_argument("-err", "--errorLogFile", help="adress to an error log text-file to append to the text body of "
                                                   "the email")
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument("-start", "--program_start", action="store_true",
                   help="use this if it is executed before the actual program starts")
group.add_argument("-fin", "--program_finished", action="store_true",
                   help="use this if it is executed after the actual program has finished")
args = parser.parse_args()
# </editor-fold>


# TODO add custom subject?
#  add info about executed file (name, purpose, hyperparameters)
separator = "\n####################################################################################################\n"
# text = ""

if args.program_start:
    text = f"Program started: {datetime.now().strftime('%d/%m/%Y, %H:%M h')}"

elif args.program_finished:
    text = f"Program finished: {datetime.now().strftime('%d/%m/%Y, %H:%M h')}"
    # add stdout log file to text/mail
    if args.outputLogFile:
        try:
            with open(args.outputLogFile, "r") as f:
                log_file = f.read().encode("ascii", "ignore").decode("ascii")

        except FileNotFoundError:
            log_file = f"Log File with name {args.outputLogFile} not found!\n"

        text += f"{separator}Output Log-File: \n\n{log_file}\n"
    # add error log file to text/mail
    if args.errorLogFile:
        try:
            with open(args.errorLogFile, "r") as f:
                error_log = f.read().encode("ascii", "ignore").decode("ascii")
        except FileNotFoundError:
            error_log = f"Log File with name {args.errorLogFile} not found!\n"
        text += f"{separator}Error Log-File: \n\n{error_log}\n"



send_mail("michael.reiter94@gmail.com", text=text, subject="Slurm Notification")
# print(text)
