import tkinter as tk
from tkinter import ttk

class ApiKeyWindow(tk.Toplevel):
    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        self.title("Enter API Key")
        self.geometry("300x150")

        self.label = ttk.Label(self, text="Please enter your Alpha Vantage API Key:")
        self.label.pack(pady=10)

        self.api_key_entry = ttk.Entry(self, show="*")
        self.api_key_entry.pack(pady=10)

        self.submit_button = ttk.Button(self, text="Submit", command=self.submit_api_key)
        self.submit_button.pack(pady=10)

    def submit_api_key(self):
        api_key = self.api_key_entry.get()
        if api_key:
            self.parent.set_api_key(api_key)
            self.destroy()
        else:
            tk.messagebox.showerror("Error", "Please enter a valid API key.")