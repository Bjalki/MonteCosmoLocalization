import os
import time
import shutil
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

def convert_md2html(download_dir:str,
                    md_filename:str='README.md',
                    outfile:str='index.html') -> None:
    '''
    Args:
        md_filename: name of file to be converted (filename.md)
        outfile: name of output file (filename.html)
        download_dir: the directory that file will be downloaded (usually "Downloads" if using Chrome)
    Return:
        None - created an html file with name from outfile
    '''
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))
    driver.implicitly_wait(1.0)
    driver.maximize_window()
    driver.get("https://codebeautify.org/markdown-to-html")

    # Upload file
    file_input = driver.find_element(by=By.ID, value='fileInput')
    driver.implicitly_wait(5)
    file_input.send_keys(os.path.abspath(md_filename))
    driver.implicitly_wait(5)
    time.sleep(2)
    # Download file
    download_btn = driver.find_element(by=By.XPATH, value='//button[@class="button is-fullwidth "]')
    download_btn.click()
    time.sleep(2)
    # Quit the browser
    driver.quit()
    print('File downloaded')

    # this is the default filename from the website
    filename = 'markdown-to-html.md'
    # Move file to current dir
    saving_dir = os.path.dirname(os.path.abspath(md_filename))
    shutil.move(os.path.join(download_dir, filename), saving_dir)
    # write finished version of html:5
    write_html(filename, outfile)
    # clean obsolete files
    if os.path.exists(filename): os.remove(filename)
    print('Done generating index.html')

def write_html(inname, outname):
    ''' To write the finished html file based on html:5 format
    Args:
        inname: name of file to be read
        outname: name of output file (filename.html)
    Return:
        None - created an html file with name from outname
    '''
    with open(inname, 'r') as f:
        lines = f.readlines()
    texts_start = ['<!DOCTYPE html>', '<html lang="en">', '<head>',
                  '<meta charset="UTF-8">', '<meta http-equiv="X-UA-Compatible" content="IE=edge">',
                  '<meta name="viewport" content="width=device-width, initial-scale=1.0">',
                  '<title>CS 371 MCL CozmoBot</title>', '</head>', '<body>']
    texts_end = ['</body>', '</html>']
    with open(outname, 'w') as f:
        f.writelines('\n'.join(texts_start))
        f.writelines(lines)
        f.writelines('\n'.join(texts_end))
    print('Done writing html:5 file')

if __name__ == '__main__':
    convert_md2html(download_dir='D:/Downloads-D')#'/Accounts/turing/students/s24/nguyqu03/Downloads'    
    