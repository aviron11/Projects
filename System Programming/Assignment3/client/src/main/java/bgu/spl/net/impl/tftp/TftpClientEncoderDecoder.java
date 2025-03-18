package bgu.spl.net.impl.tftp;

import java.util.Arrays;

import bgu.spl.net.api.MessageEncoderDecoder;

public class TftpClientEncoderDecoder implements MessageEncoderDecoder<byte[]> {

    private byte[] buffer = new byte[1];
    private byte[] data = new byte[1];
    int length = 0;
    int packetSize = 0;
    private short opCode = 0;

    private byte[] response;

    @Override
    public byte[] decodeNextByte(byte nextByte) {
        if(length >= buffer.length){
            buffer = Arrays.copyOf(buffer, length*2);
        }

        buffer[length++] = nextByte;

        if(length == 2){
            opCode = getShort(Arrays.copyOfRange(buffer, 0, 2));
        }

        if(opCode == 3){
            if(length == 4){
                packetSize = getShort(Arrays.copyOfRange(buffer, 2, 4));
            }
            if(length == 6 + packetSize){
                data = Arrays.copyOfRange(buffer, 0, length);
                length = 0;
                opCode = 0;
                return data;
            }
        }

        if(opCode == 4){
            if(length == 4){
                data = Arrays.copyOfRange(buffer, 0, length);
                length = 0;
                opCode = 0;
                return data;
            }
        }

        if(opCode == 5){
            if(length > 4){
                if(nextByte == 0){
                    data = Arrays.copyOfRange(buffer, 0, length - 1);
                    length = 0;
                    opCode = 0;
                    return data;
                }
            }
        }

        if(opCode == 9){
            if(length > 3){
                if(nextByte == 0){
                    data = Arrays.copyOfRange(buffer, 0, length - 1);
                    length = 0;
                    opCode = 0;
                    return data;
                }
            }
        }

        return null;
    }

    private short getShort(byte[] copyOfRange) {
        short ans = (short) ((short) (copyOfRange[0] & 0xFF) << 8 | (short)(copyOfRange[1] & 0xFF));
        return ans;
    }

    @Override
    public byte[] encode(byte[] message) {
        //convert the message to a string using UTF-8 encoding
        String str = new String(message, 0, message.length, java.nio.charset.StandardCharsets.UTF_8);
        String[] words = str.split(" ");
        if(words[0].equals("RRQ")){
            byte[] file = str.substring(words[0].length() + 1).getBytes();
            response = new byte[file.length + 3];
            response[0] = 0;
            response[1] = (byte) 1;
            for(int i = 0; i < file.length; i++){
                response[i+2] = file[i];
            }
        }
        else if(words[0].equals("WRQ")){
            byte[] file = str.substring(words[0].length() + 1).getBytes();
            response = new byte[file.length + 3];
            response[0] = 0;
            response[1] = (byte) 2;
            for(int i = 0; i < file.length; i++){
                response[i+2] = file[i];
            }
        }
        else if(words[0].equals("DIRQ")){
            response = new byte[2];
            response[0] = 0;
            response[1] = 6;
        }
        else if(words[0].equals("LOGRQ")){
            response = new byte[words[1].length() + 3];
            response[0] = 0;
            response[1] = 7;
            for(int i = 0; i < words[1].length(); i++){
                response[i+2] = (byte) words[1].charAt(i);
            }
        }
        else if(words[0].equals("DELRQ")){
            byte[] file = str.substring(words[0].length() + 1).getBytes();
            response = new byte[file.length + 3];
            response[0] = 0;
            response[1] = 8;
            for(int i = 0; i < file.length; i++){
                response[i+2] = file[i];
            }
        }
        else if(words[0].equals("DISC")){
            response = new byte[2];
            response[0] = 0;
            response[1] = 10;
        }
        byte [] temp = response;
        response = null;
        return temp;
    }
}
