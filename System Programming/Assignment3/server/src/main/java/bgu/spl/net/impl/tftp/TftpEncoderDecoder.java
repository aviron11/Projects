package bgu.spl.net.impl.tftp;

import java.util.Arrays;

import bgu.spl.net.api.MessageEncoderDecoder;

public class TftpEncoderDecoder implements MessageEncoderDecoder<byte[]> {
    //TODO: Implement here the TFTP encoder and decoder

    private byte[] buffer = new byte[1];
    private byte[] data = new byte[1];
    int length = 0;
    int packetSize = 0;
    private short opCode = 0;


    @Override
    public byte[] decodeNextByte(byte nextByte) {
        if(length >= buffer.length){
            buffer = Arrays.copyOf(buffer, length*2);
        }

        buffer[length++] = nextByte;

        if(length == 2){
            opCode = getShort(Arrays.copyOfRange(buffer, 0, 2));
        }

        if(opCode == 1 || opCode == 2 || opCode == 7 || opCode == 8){
            if(nextByte == 0){
                data = Arrays.copyOfRange(buffer, 0, length -1);
                length = 0;
                opCode = 0;
                return data;
            }
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

        if(opCode == 6 ||opCode == 10){
            data = Arrays.copyOfRange(buffer, 0, length);
            length = 0;
            opCode = 0;
            return data;
        }

        return null;
    }

    private short getShort(byte[] copyOfRange) {
        short ans = (short) (((short) (copyOfRange[0] & 0xFF) << 8 | (short)(copyOfRange[1] & 0xFF)));
        return ans;
    }

    @Override
    public byte[] encode(byte[] message) {
        return message;
    }
}