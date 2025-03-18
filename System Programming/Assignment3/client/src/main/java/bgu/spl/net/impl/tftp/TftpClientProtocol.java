package bgu.spl.net.impl.tftp;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.Arrays;
import java.util.LinkedList;


public class TftpClientProtocol {

    private short opCode = 0;
    protected boolean loggedIn = true;
    private byte[] fileData;

    public String filePath = "./Files/";
    public String FilePath = "./Files";
    public String fileName = null;
    
    public int blockNumber = 0;

    private byte[][] dataPackets;
    LinkedList<byte[]> DataPacketsBuffer = new LinkedList<>();

    protected String action = null;

    public byte[] process(byte[] message) {

        opCode = getShort(Arrays.copyOfRange(message, 0, 2));

        if(opCode == 3){
            //data packet
            int temp = blockNumber;

            blockNumber = getShort(Arrays.copyOfRange(message, 4, 6));

            if(blockNumber == 0){
                //recived string of files separated by a 0 byte, need to print them
                String files = new String(message, 6, message.length - 6, StandardCharsets.UTF_8);
                String[] filesArr = files.split("\0");
                for(String file : filesArr){
                    System.out.println(file);
                }
            }

            else{
                
                byte[] data = Arrays.copyOfRange(message, 6, message.length); 
                
                DataPacketsBuffer.add(data);
                
                blockNumber = getShort(Arrays.copyOfRange(message, 4, 6));

                if(blockNumber == temp + 1){

                    try(FileOutputStream fos = new FileOutputStream(filePath + fileName, true)){
                        if(data.length < 512){
                            for(byte[] packet : DataPacketsBuffer){
                                fos.write(packet);
                            }
                            System.out.println("RRQ " + fileName + " complete");
                            this.blockNumber = 0;
                            DataPacketsBuffer = new LinkedList<>();
                            fileName = null;
                        }
                        return sendAck(temp + 1);
                    }catch(IOException e){e.printStackTrace();}
                }
                else{
                    DataPacketsBuffer = new LinkedList<>();
                    return sendError((short) 0, "Unknown error");
                }
            }
            

            
        }
        if(opCode == 4){
            //ack packet
            int temp = blockNumber;
            temp = getShort(Arrays.copyOfRange(message, 2, 4));
            if(temp == 0 && fileName != null){
                File file = new File(filePath + fileName);
                fileData = new byte[(int) file.length()];
                try(FileInputStream fis = new FileInputStream(filePath + fileName)){
                    fis.read(fileData);
                }catch(IOException e){
                    e.printStackTrace();
                }
                dataPackets = new byte[fileData.length/512 + 1][];
                for(int i = 0 ; i < fileData.length/512 ; i++){
                    dataPackets[i] = Arrays.copyOfRange(fileData, i*512, (i+1)*512);
                }
                dataPackets[fileData.length/512] = Arrays.copyOfRange(fileData, fileData.length/512 * 512, fileData.length);
                System.out.println("ACK " + blockNumber);
                blockNumber++;
                return sendData(dataPackets[0]);
            }
            else if(fileName != null && temp == dataPackets.length){
                System.out.println("WRQ " + fileName + " complete");
                fileName = null;
            }
            else if(fileName != null){
            
                if(dataPackets[temp].length < 512){
                    System.out.println("ACK " + blockNumber);
                    blockNumber++;
                    opCode = 0;
                    return sendData(dataPackets[temp]);
                }

                else if(temp == blockNumber){
                    System.out.println("ACK " + blockNumber);
                    blockNumber++;
                    return sendData(dataPackets[temp]);
                }
            }
            else if(action.equals("LOGRQ")) {
                loggedIn = true;
                System.out.println("ACK 0");
            }
            else{
                System.out.println("ACK 0");
            }
            
        }
        else if(opCode == 5){
            //error packet
            short errorCode = getShort(Arrays.copyOfRange(message, 2, 4));
            String errorMsg = new String(message, 4, message.length - 4, StandardCharsets.UTF_8);
            System.out.println("Error " + errorCode + ": " + errorMsg);
            fileName = null;
        }
        else if(opCode == 9){
            //BCast packet
            String tempFileName = new String(message, 3, message.length - 3, StandardCharsets.UTF_8);
            byte delOrAdd = message[2];
            if(delOrAdd == 0){
                System.out.println("BCAST " + tempFileName + " deleted");
            }
            else{
                System.out.println("BCAST " + tempFileName + " added");
            }
        }
        return null;
    }

    private short getShort(byte[] copyOfRange) {
        short ans = (short) (short) ((copyOfRange[0] & 0xff) << 8 | (copyOfRange[1] & 0xff));;
        return ans;
    }

    public boolean shouldTerminate() {
        return !loggedIn;
    }

    public byte[] sendAck(int blockNumber) {
        byte[] ack = new byte[4];
        ack[0] = 0;
        ack[1] = 4;
        byte[] blocknumberAsBytes = new byte[]{(byte) (blockNumber >> 8), (byte) (blockNumber & 0xff)};
        ack[2] = blocknumberAsBytes[0];
        ack[3] = blocknumberAsBytes[1];
        return ack;
    }

    public byte[] sendError(short errorCode, String errorMsg) {
        byte[] error = new byte[5 + errorMsg.length()];
        error[0] = 0;
        error[1] = 5;
        byte[] errorCodeAsBytes = new byte[]{(byte) (errorCode >> 8), (byte) (errorCode & 0xff)};
        error[2] = errorCodeAsBytes[0];
        error[3] = errorCodeAsBytes[1];
        byte[] msg = errorMsg.getBytes();
        for (int i = 0; i < msg.length; i++) {
            error[i + 4] = msg[i];
        }
        error[4 + msg.length] = 0;
        return error;
    }

    private byte[] sendData(byte[] data){
        byte[] dataPacket = new byte[data.length + 6];
        dataPacket[0] = 0;
        dataPacket[1] = 3;
        byte[] dataLengthAsBytes = new byte[]{(byte) (data.length >> 8), (byte) (data.length & 0xff)};
        dataPacket[2] = dataLengthAsBytes[0];
        dataPacket[3] = dataLengthAsBytes[1];
        byte[] blocknumberAsBytes = new byte[]{(byte) (blockNumber >> 8), (byte) (blockNumber & 0xff)};
        dataPacket[4] = blocknumberAsBytes[0];
        dataPacket[5] = blocknumberAsBytes[1];
        if(data.length < 512){
            this.blockNumber = 0;
        }
        for(int i = 0; i < data.length; i++){
            dataPacket[i + 6] = data[i];
        }
        return dataPacket;
    }

    public boolean checkValid(String[] words, String line){
        if(words[0].equals("RRQ") || words[0].equals("WRQ") || words[0].equals("LOGRQ") || words[0].equals("DELRQ")){
            if (words.length == 1){
                System.out.println("Missing argument");
                return false;
            }  
            else if(words[0].equals("RRQ")){
                String fileName = line.substring(words[0].length() + 1);
                File folder = new File(FilePath);
                File[] files = folder.listFiles();
                for(File file : files){
                    if (fileName.equals(file.getName())){
                        System.out.println("File already exists");
                        return false;
                    }
                }
            }
            else if (words[0].equals("WRQ")){
                String fileName = line.substring(words[0].length() + 1);
                File folder = new File(FilePath);
                File[] files = folder.listFiles();
                for(File file : files){
                    if (fileName.equals(file.getName())){
                        return true;
                    }
                }
                System.out.println("File doesn't exist");
                return false;
            }
        }
        return true;
    }

}