package bgu.spl.net.impl.tftp;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.net.Socket;
import java.util.NoSuchElementException;
import java.util.Scanner;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.Semaphore;
import java.io.IOException;


public class TftpClient {
    
    private static BlockingQueue<byte[]> messages = new LinkedBlockingQueue<>();
    private static Semaphore s = new Semaphore(1, true);

    public static void main(String[] args) {
        if (args.length == 0) {
            args = new String[]{"localhost"};
        }

        TftpClientEncoderDecoder encdec = new TftpClientEncoderDecoder();
        TftpClientProtocol protocol = new TftpClientProtocol();

        System.out.println("client started");

        try (Socket sock = new Socket(args[0], 7777);
                BufferedInputStream in = new BufferedInputStream(sock.getInputStream());
                BufferedOutputStream out = new BufferedOutputStream(sock.getOutputStream());
                Scanner scanner = new Scanner(System.in);) {
                

            Thread keyboardThread = new Thread(() -> {
                try{
                    String line = null;
                    do {
                        try{
                            line = scanner.nextLine();
                        }catch(NoSuchElementException e){
                            System.out.println("Client has disconnected");
                            break;
                        }
                        byte[] ans = null;
                        String words[] = line.split(" ");
                        if(protocol.checkValid(words, line)){
                            ans = encdec.encode(line.getBytes());
                            if(ans == null){
                                System.out.println("Invalid command");
                                continue;
                            }
                            protocol.action = words[0];
                            if(words[0].equals("RRQ") || words[0].equals("WRQ")){
                                protocol.fileName = line.substring(words[0].length() + 1);
                            }
                            if (s.tryAcquire()){
                                out.write(ans);
                                out.flush();
                                s.release();
                            }
                            else{
                                messages.add(ans);
                            }
                        } 
                        while (!messages.isEmpty()){
                                s.tryAcquire();
                                out.write(messages.poll());
                                out.flush();
                                s.release();
                        }
                    } while(!protocol.shouldTerminate() && !line.equals("DISC"));
                }catch(IOException | IllegalStateException e){
                    e.printStackTrace();
                }
            });

            Thread listeningThread = new Thread(() -> {
                int read = 0;
                try {
                    while ((read = in.read()) >= 0 && !protocol.shouldTerminate()) {
                        byte[] nextMessage = encdec.decodeNextByte((byte) read);
                        if (nextMessage != null) {
                            byte[] ans = protocol.process(nextMessage);
                            if (ans != null) {
                                try{
                                    s.acquire();
                                    out.write(ans);
                                    out.flush();
                                    s.release();
                                }catch(InterruptedException e){}
                            }
                        }
                    }
                } catch (IOException e) {
                    e.printStackTrace();
                }                
            });

            keyboardThread.start();
            listeningThread.start();

            keyboardThread.join(); // Wait for keyboard thread to finish
            listeningThread.join(); // Wait for listening thread to finish

        }catch(IOException | InterruptedException e){
            e.printStackTrace();
        }
    }
}