import smtplib



def send_mail_fun(receiver_mail, sender_mail, sender_password, subject, msg):
    try:
        # Set the server and port
        server = smtplib.SMTP('smtp.gmail.com', 587)
        # Start TLS encryption
        server.starttls()
        # Login to the server
        server.login(sender_mail, sender_password)
        # Send the email
        to = receiver_mail
        subject = subject
        body = msg
        message = f"Subject: {subject}\n\n{body}"
        server.sendmail(sender_mail, to, message)
        # Close the server
        server.quit()
        print("mail send succesfully >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        return "Mail succesfully sent."
    except Exception as e:
        print(f"get error in sending mail: {str(e)}>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        return "Mail not sent."

    
def send_error_mail(receiver_mail, sender_mail, sender_password):
    subject="error occurred"
    msg = "something went wrong with server please try after some time."
    response = send_mail_fun(receiver_mail, sender_mail, sender_password, subject, msg)
    return response
    
    
def send_complete_mail(receiver_mail, sender_mail, sender_password):
    subject = "train successfull."
    msg = "Model trainning is successfully done, Your model is ready."
    response = send_mail_fun(receiver_mail, sender_mail, sender_password, subject, msg)
    return response
    
# receiver_mail = "jaydevsinh.kshatrainfotech@gmail.com"
# sender_mail = "viral.kshatrainfotech@gmail.com"
# sender_password = "nurbxmxtyovnqgsn"
# status = send_mail(receiver_mail, sender_mail, sender_password)