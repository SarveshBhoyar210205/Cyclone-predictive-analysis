import webbrowser
import time

def open_html_file():
    # Wait for a few seconds to ensure Flask server is up
    time.sleep(5)
    # Open the HTML file
    webbrowser.open(r'C:\Users\Dell\Desktop\H5\flask\location.html')

if __name__ == '__main__':
    open_html_file()
