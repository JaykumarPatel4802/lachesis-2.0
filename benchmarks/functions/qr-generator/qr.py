import qrcode
from time import time

def generate_qr(url):
    start = time()
    qr = qrcode.QRCode(version = 1,
                   box_size = 20,
                   border = 2)
    # Adding data to the instance 'qr'
    qr.add_data(url)
    
    qr.make(fit = True)
    img = qr.make_image(fill_color = 'red',
                        back_color = 'white')
    img.save('MyQRCode2.png')
    latency = time() - start
    return latency


def main(params):
    url = params['url']
    latency = generate_qr(url)
    ret_val = {}
    ret_val["latency"] = latency
    return ret_val