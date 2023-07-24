import os.path
import datetime
import pickle
import subprocess
import tkinter as tk
import cv2
from PIL import Image, ImageTk
import face_recognition
import util

class App:
    def __init__(self):
        self.main_window = tk.Tk()
        self.main_window.geometry("1200x520+350+100")

        self.db_dir = './db'
        if not os.path.exists(self.db_dir):
            os.mkdir(self.db_dir)

        self.log_path = './log.txt'

        self.initialize_widgets()

    def initialize_widgets(self):
        self.login_button_main_window = util.get_button(self.main_window, 'login', 'green', self.login)
        self.login_button_main_window.place(x=750, y=200)

        self.logout_button_main_window = util.get_button(self.main_window, 'logout', 'red', self.logout)
        self.logout_button_main_window.place(x=750, y=300)

        self.register_new_user_button_main_window = util.get_button(self.main_window, 'register new user', 'gray',
                                                                    self.register_new_user, fg='black')
        self.register_new_user_button_main_window.place(x=750, y=400)

        self.webcam_label = util.get_img_label(self.main_window)
        self.webcam_label.place(x=10, y=0, width=700, height=500)

        self.add_webcam(self.webcam_label)

    def add_webcam(self, label):
        if 'cap' not in self.__dict__:
            self.cap = cv2.VideoCapture(0)

        self._label = label
        self.process_webcam()

    def process_webcam(self):
        ret, frame = self.cap.read()

        self.most_recent_capture_arr = frame
        img_ = cv2.cvtColor(self.most_recent_capture_arr, cv2.COLOR_BGR2RGB)
        self.most_recent_capture_pil = Image.fromarray(img_)
        imgtk = ImageTk.PhotoImage(image=self.most_recent_capture_pil)
        self._label.imgtk = imgtk
        self._label.configure(image=imgtk)

        self._label.after(20, self.process_webcam)

    def login(self):
        unknown_img_path = "./.tmp.jpg"
        cv2.imwrite(unknown_img_path, self.most_recent_capture_arr)

        unknown_encoding = face_recognition.face_encodings(face_recognition.load_image_file(unknown_img_path))

        if not unknown_encoding:
            util.msg_box('Ups...', 'No face detected. Please try again.')
            return

        unknown_encoding = unknown_encoding[0]

        found_name = None
        for file_name in os.listdir(self.db_dir):
            if file_name.endswith(".pickle"):
                with open(os.path.join(self.db_dir, file_name), 'rb') as f:
                    known_encoding = pickle.load(f)
                    if face_recognition.compare_faces([known_encoding], unknown_encoding)[0]:
                        found_name = file_name[:-len(".pickle")]
                        break

        if found_name:
            util.msg_box('Welcome back!', 'Welcome, {}.'.format(found_name))
            with open(self.log_path, 'a') as f:
                f.write('{},{},in\n'.format(found_name, datetime.datetime.now()))
                f.close()
        else:
            util.msg_box('Ups...', 'Unknown user. Please register a new user or try again.')

    def logout(self):
        if self.logged_in_user:
            with open(self.log_path, 'a') as f:
                f.write('{},{},out\n'.format(self.logged_in_user, datetime.datetime.now()))
            util.msg_box('Goodbye!', 'Logged out successfully.')
            self.logged_in_user = None
        else:
            util.msg_box('Hey!', 'You are not logged in.')





    def register_new_user(self):
        self.register_new_user_window = tk.Toplevel(self.main_window)
        self.register_new_user_window.geometry("1200x520+370+120")

        self.accept_button_register_new_user_window = util.get_button(self.register_new_user_window, 'Accept', 'green', self.accept_register_new_user)
        self.accept_button_register_new_user_window.place(x=750, y=300)

        self.try_again_button_register_new_user_window = util.get_button(self.register_new_user_window, 'Try again', 'red', self.try_again_register_new_user)
        self.try_again_button_register_new_user_window.place(x=750, y=400)

        self.capture_label = util.get_img_label(self.register_new_user_window)
        self.capture_label.place(x=10, y=0, width=700, height=500)

        self.add_img_to_label(self.capture_label)

        self.entry_text_register_new_user = util.get_entry_text(self.register_new_user_window)
        self.entry_text_register_new_user.place(x=750, y=150)

        self.text_label_register_new_user = util.get_text_label(self.register_new_user_window, 'Please, \ninput username:')
        self.text_label_register_new_user.place(x=750, y=70)

    def accept_register_new_user(self):
        name = self.entry_text_register_new_user.get(1.0, "end-1c")

        embeddings = face_recognition.face_encodings(self.register_new_user_capture)
        if not embeddings:
            util.msg_box('Error!', 'No face detected. Try again.')
            return

        embeddings = embeddings[0]

        file_path = os.path.join(self.db_dir, '{}.pickle'.format(name))
        with open(file_path, 'wb') as file:
            pickle.dump(embeddings, file)

        util.msg_box('Success!', 'User was registered successfully!')
        self.register_new_user_window.destroy()

    def try_again_register_new_user(self):
        self.register_new_user_window.destroy()

    def add_img_to_label(self, label):
        imgtk = ImageTk.PhotoImage(image=self.most_recent_capture_pil)
        label.imgtk = imgtk
        label.configure(image=imgtk)

        self.register_new_user_capture = self.most_recent_capture_arr.copy()

    def start(self):
        self.main_window.mainloop()

if __name__ == "__main__":
    app = App()
    app.start()










