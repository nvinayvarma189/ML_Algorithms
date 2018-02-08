import paho.mqtt.client as mqtt
import ssl

def onc(c,userdata,flags,rc):
    print('connected to aamazon with rc '+str(rc))
    c.subscribe('mytopic/today')

def onm(c,userdata,msg):
    print(msg.payload.decode())
    x=msg.payload.decode()
    if x=='hi':
        c.publish("mytopic/today","hello from python from TechTrunk")

c=mqtt.Client()
rootca=r'C:\Users\Robot\Desktop\aws\rootca.crt'
crtfile=r'C:\Users\Robot\Desktop\aws\a5affe1bba-certificate.pem.crt'
key=r'C:\Users\Robot\Desktop\aws\a5affe1bba-private.pem.key'
c.tls_set(rootca,certfile=crtfile,keyfile=key,
          cert_reqs=ssl.CERT_REQUIRED,
          tls_version=ssl.PROTOCOL_TLSv1_2,
          ciphers=None)

c.connect('a16wvf8spsli06.iot.us-west-2.amazonaws.com',8883,60)

c.on_connect=onc
c.on_message=onm
c.loop_forever()
