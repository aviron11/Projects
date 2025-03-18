package bgu.spl.net.impl.tftp;

import java.io.File;
import java.nio.charset.StandardCharsets;

public class Testing {

    public static void main(String[] args) {
        byte[] message = "ggA.txt".getBytes();
        String fileName = new String(message, 2, message.length - 2, StandardCharsets.UTF_8);
        String filePath = "C:\\Users\\avraa\\OneDrive\\Desktop\\VS projects\\Java\\Assignment3_final\\Skeleton\\server\\Flies\\";
        File file = new File(filePath + fileName);
        System.out.println(file.exists());

        short a = 10;
        byte[] a_bytes = new byte[]{(byte) (a >> 8), (byte) (a & 0xff)};
        short b = getShort(a_bytes);
        System.out.println(b);


    }

    public static short getShort(byte[] bytes) {
        return (short) ((bytes[0] & 0xff) << 8 | (bytes[1] & 0xff));
    }
    
}
