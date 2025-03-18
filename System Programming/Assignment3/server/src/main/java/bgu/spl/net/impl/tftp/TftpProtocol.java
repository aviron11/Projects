package bgu.spl.net.impl.tftp;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.Arrays;
import java.util.LinkedList;
import java.util.concurrent.ConcurrentHashMap;

import bgu.spl.net.api.BidiMessagingProtocol;
import bgu.spl.net.srv.Connections;

class holder{
    static ConcurrentHashMap<Integer, Boolean> ids_login = new ConcurrentHashMap<>();
}

public class TftpProtocol implements BidiMessagingProtocol<byte[]>  {

    private short opCode = 0;
    private boolean loggedIn = false;
    private byte[] fileData;

    private String filePath = "./Files/";

    private String fileName = null;
    
    public short blockNumber = 0;

    private byte[][] dataPackets;
    private boolean shouldTerminate = false;

    private Connections<byte[]> connections;
    private int connectionId;

    private LinkedList<byte[]> dataPacketsBuffer = new LinkedList<>();

    private String userName = null;

    @Override
    public void start(int connectionId, Connections<byte[]> connections) {
        this.shouldTerminate = false;
        this.connectionId = connectionId;
        this.connections = connections;
    }

    @Override
    public void process(byte[] message) {

        opCode = getShort(Arrays.copyOfRange(message, 0, 2));

        if(opCode < 0 || opCode > 10){
            sendError((short) 4, "Unknown opcode");
            blockNumber = 0;
            opCode = 0;
            fileName = null;
        }

        else if(opCode == 7){
            if(loggedIn){
                sendError((short) 7, "User already logged in");
                blockNumber = 0;
                opCode = 0;
                fileName = null;
            }
            else{
                userName = new String(message, 2, message.length - 2, StandardCharsets.UTF_8);
                if(!connections.userExists(userName)){
                    holder.ids_login.put(connectionId, true);
                    connections.addUser(connectionId, userName);
                    loggedIn = true;
                    sendAck(0);
                }
                else{
                    sendError((short) 7, "User name already in use");
                }
                
            }
        }

        else if(loggedIn){
            if(opCode == 1){

                blockNumber = 1;

                // opening the file 
                fileName = new String(message, 2, message.length - 2, StandardCharsets.UTF_8);
                File file = new File(filePath + fileName);

                // reading the file and separating it to 512 byte packets
                if(file.exists()){

                    long fileLength = file.length();

                    fileData = new byte[(int) fileLength];

                    try(FileInputStream fis = new FileInputStream(filePath + fileName);){

                        fis.read(fileData);

                        //divide fileData to 512 byte packets

                        dataPackets = new byte[fileData.length/512 + 1][];

                        for(int i = 0 ; i < fileData.length/512 ; i++){
                            dataPackets[i] = Arrays.copyOfRange(fileData, i*512, (i+1)*512);
                        }

                        dataPackets[fileData.length/512] = Arrays.copyOfRange(fileData, fileData.length/512 * 512, fileData.length);

                        sendData(dataPackets[0]);

                    }catch(IOException e){e.printStackTrace();}
                }
                else{
                    sendError((short) 1, "File not found");
                    blockNumber = 0;
                    opCode = 0;
                    fileName = null;
                }
            }
                    
            else if(opCode == 2){

                fileName = new String(message, 2, message.length - 2, StandardCharsets.UTF_8);
                
                File file = new File(filePath + fileName);

                if(!file.exists()){
                    blockNumber = 0;
                    sendAck(blockNumber);
                    blockNumber++;
                }
                else{
                    sendError((short) 5, "File already exists");
                    blockNumber = 0;
                    opCode = 0;
                    fileName = null;
                }
            }

            else if(opCode == 3){
                //data packet

                int temp = blockNumber;

                blockNumber = getShort(Arrays.copyOfRange(message, 4, 6));

                if(blockNumber == temp){

                    byte[] data = Arrays.copyOfRange(message, 6, message.length);   
                            
                    dataPacketsBuffer.add(data);

                    sendAck(blockNumber);
                    blockNumber++;

                    if(data.length < 512){
                        try(FileOutputStream fos = new FileOutputStream(filePath + fileName)){
                            for(byte[] dataPacket : dataPacketsBuffer){
                                fos.write(dataPacket);
                            }
                        }catch(IOException e){e.printStackTrace();}

                        sendBCast(fileName, 1);
                        
                        blockNumber = 0;
                        opCode = 0;
                        fileName = null;
                        dataPacketsBuffer = new LinkedList<>();
                    }
                    
                }
                else{
                    sendError((short) 0, "Unknown error");
                    blockNumber = 0;
                    opCode = 0;
                    fileName = null;
                    dataPacketsBuffer = new LinkedList<>();
                }
            }

            else if(opCode == 4){
                //ack packet

                int temp = getShort(Arrays.copyOfRange(message, 2, 4));

                if(dataPackets[temp - 1].length < 512){
                    blockNumber = 0;
                    opCode = 0;
                    fileName = null;
                    dataPacketsBuffer = new LinkedList<>();
                }

                else if(temp == blockNumber){
                    blockNumber++;
                    sendData(dataPackets[temp]);
                }
                else{
                    sendError((short) 0, "Unknown error");
                    blockNumber = 0;
                    opCode = 0;
                    fileName = null;
                    dataPacketsBuffer = new LinkedList<>();
                }
            }

            else if(opCode == 6){
                //dirq
                File folder = new File(filePath);
                File[] listOfFiles = folder.listFiles();
                String fileNames = "";
                for(int i = 0; i < listOfFiles.length; i++){
                    fileNames += listOfFiles[i].getName() + '\0';
                }
                sendData(fileNames.getBytes(StandardCharsets.UTF_8));
            }

            else if(opCode == 8){
                //delete file
                fileName = new String(message, 2, message.length - 2, StandardCharsets.UTF_8);
                File file = new File(filePath + fileName);
                if(file.exists()){
                    file.delete();
                    sendAck(0);
                    sendBCast(fileName, 0);
                    blockNumber = 0;
                    opCode = 0;
                    fileName = null;
                }
                else{
                    sendError((short) 1, "File not found");
                    blockNumber = 0;
                    opCode = 0;
                    fileName = null;
                }
            }

            else if(opCode == 10){
                //disc
                disconnect();
            }

            else{
                sendError((short) 4, "Unknown opcode");
                blockNumber = 0;
                opCode = 0;
                fileName = null;
            }

        }
        else{
            if(opCode == 1 || opCode == 6 || opCode == 8){ //rrq, dirq or delrq
                sendError((short) 6, "Access violation File cannot be written, read or deleted");
                blockNumber = 0;
                opCode = 0;
                fileName = null;
            } 
            else if(opCode == 2){ //wrq
                fileName = new String(message, 2, message.length - 2, StandardCharsets.UTF_8);
                File file = new File(filePath + fileName);
                if(file.exists()){
                    sendError((short) 5, "File already exists - File name exists on WRQ.");
                }
                else{
                    sendError((short) 6, "Access violation File cannot be written, read or deleted");
                }
                blockNumber = 0;
                opCode = 0;
                fileName = null;
            }
            else{
                sendError((short) 6, "User not logged in");
                blockNumber = 0;
                opCode = 0;
                fileName = null;
            }
        }
    
    }

    private void disconnect() {
        sendAck(0);
        shouldTerminate = true;
        connections.disconnect(connectionId);
        holder.ids_login.remove(connectionId);
        connections.removeUser(connectionId);
    }

    @Override
    public boolean shouldTerminate() {
        return shouldTerminate;
    } 

    private short getShort(byte[] copyOfRange) {
        short ans = (short) (short) ((copyOfRange[0] & 0xff) << 8 | (copyOfRange[1] & 0xff));;
        return ans;
    }

    private void sendAck(int blockNumber){
        byte[] ack = new byte[4];
        ack[0] = 0;
        ack[1] = 4;
        byte[] blocknumberAsBytes = new byte[]{(byte) (blockNumber >> 8), (byte) (blockNumber & 0xff)};
        ack[2] = blocknumberAsBytes[0];
        ack[3] = blocknumberAsBytes[1];
        connections.send(connectionId, ack);
    }

    private void sendError(short errorCode, String errorMsg){
        byte[] error = new byte[errorMsg.length() + 5];
        error[0] = 0;
        error[1] = 5;
        byte[] errorCodeAsBytes = new byte[]{(byte) (errorCode >> 8), (byte) (errorCode & 0xff)};
        error[2] = errorCodeAsBytes[0];
        error[3] = errorCodeAsBytes[1];
        byte[] errorMsgBytes = errorMsg.getBytes(StandardCharsets.UTF_8);
        for(int i = 0; i < errorMsgBytes.length; i++){
            error[i + 4] = errorMsgBytes[i];
        }
        error[error.length - 1] = 0;
        connections.send(connectionId, error);
    }

    private void sendBCast(String filename, int delOrAdd){
        byte[] filenameBytes = filename.getBytes(StandardCharsets.UTF_8);
        byte[] bcast = new byte[3 + filenameBytes.length + 1];
        bcast[0] = 0;
        bcast[1] = 9;
        bcast[2] = (byte) delOrAdd;
        for(int i = 0; i < filenameBytes.length; i++){
            bcast[i + 3] = filenameBytes[i];
        }
        bcast[bcast.length - 1] = 0;
        for(Integer id : holder.ids_login.keySet()){
            connections.send(id, bcast);
        }
    }

    private void sendData(byte[] data){
        byte[] dataPacket = new byte[data.length + 6];
        dataPacket[0] = 0;
        dataPacket[1] = 3;
        byte[] dataLengthAsBytes = new byte[]{(byte) (data.length >> 8), (byte) (data.length & 0xff)};
        dataPacket[2] = dataLengthAsBytes[0];
        dataPacket[3] = dataLengthAsBytes[1];
        byte[] blocknumberAsBytes = new byte[]{(byte) (blockNumber >> 8), (byte) (blockNumber & 0xff)};
        dataPacket[4] = blocknumberAsBytes[0];
        dataPacket[5] = blocknumberAsBytes[1];
        for(int i = 0; i < data.length; i++){
            dataPacket[i + 6] = data[i];
        }
        connections.send(connectionId, dataPacket);
    }
    
}