import smtplib


def send_mail(to_address, text, subject="This is no spam"):
    gmail = "hundreddaysofcodemr@gmail.com"
    # noinspection SpellCheckingInspection
    gmail_app_pw = "wgqvfuvilcsplgdb"
    with smtplib.SMTP("smtp.gmail.com", port=587) as connection:
        connection.starttls()  # encryption
        connection.login(user=gmail, password=gmail_app_pw)
        connection.sendmail(from_addr=gmail,
                            to_addrs=to_address,
                            msg=f"Subject:{subject}\n\n{text}")


def jupyter_setup():
    import os

    # change working directory to main folder
    root_dir = "python"
    _, current_folder = os.path.split(os.getcwd())
    if current_folder != root_dir:
        os.chdir("../")

    # add path variables to avoid "audio module not found"-error
    path = os.environ.get("PATH")
    min_additional_path = "C:\\Users\\Michi\\Anaconda3\\envs\\python_v3-8\\Library\\bin;" \
                          "C:\\Users\\micha\\anaconda3\\envs\\ai38\\Library\\bin;"
    combined_path = min_additional_path + path
    os.environ["PATH"] = combined_path
