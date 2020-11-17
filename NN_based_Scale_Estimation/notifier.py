'''
Email-based ML Learning Status Notifier Module

Python SMTP email reference : http://pythonstudy.xyz/python/article/508-%EB%A9%94%EC%9D%BC-%EB%B3%B4%EB%82%B4%EA%B8%B0-SMTP
Outlook SMTP Setting : https://support.microsoft.com/ko-kr/office/outlook-com%EC%9D%98-pop-imap-%EB%B0%8F-smtp-%EC%84%A4%EC%A0%95-d088b986-291d-42b8-9564-9c414e2aa040
'''

import smtplib
from email.mime.text import MIMEText

class notifier_Outlook:

    def __init__(self, sender_email='', sender_email_pw=''):

        self.sender_email = sender_email
        self.sender_email_pw = sender_email_pw

        print('Notifier [Sender : {}]'.format(self.sender_email))

    def send(self, receiver_email='', title= '', contents=''):

        if (self.sender_email != '') and (self.sender_email_pw != ''):

            smtp = smtplib.SMTP('smtp.office365.com', 587)
            smtp.ehlo()
            smtp.starttls()
            smtp.login(self.sender_email, self.sender_email_pw)

            msg = MIMEText(contents)
            msg['Subject'] = title
            msg['To'] = receiver_email

            smtp.sendmail(self.sender_email, receiver_email, msg.as_string())

            print('Sending results to {}'.format(receiver_email))

            smtp.quit()