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
