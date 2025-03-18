package bgu.spl.net.srv;

import bgu.spl.net.api.MessageEncoderDecoder;
import bgu.spl.net.api.BidiMessagingProtocol;
import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.IOException;
import java.net.Socket;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.Semaphore;

public class BlockingConnectionHandler<T> implements Runnable, ConnectionHandler<T> {

    private final BidiMessagingProtocol<T> protocol;
    private final MessageEncoderDecoder<T> encdec;
    private final Socket sock;
    private BufferedInputStream in;
    private BufferedOutputStream out;
    private volatile boolean connected = true;
    private ConcurrentLinkedQueue<T> taskQueue;
    private Connections<T> connections;
    private int connectionId;
    Semaphore semaphore;
    

    public BlockingConnectionHandler(
        Socket sock,
        MessageEncoderDecoder<T> reader, 
        BidiMessagingProtocol<T> protocol,
        ConnectionsImpl<T> connections,
        int connectionId) 
    {
        this.sock = sock;
        this.encdec = reader;
        this.protocol = protocol;
        this.connections = connections;
        this.connectionId = connectionId;
        this.protocol.start(connectionId, connections);
        this.taskQueue = new ConcurrentLinkedQueue<>(); 
        this.semaphore = new Semaphore(1, true);
    }

    @Override
    public void run() {
        try (Socket sock = this.sock) { //just for automatic closing
            int read = -1;

            in = new BufferedInputStream(sock.getInputStream());
            out = new BufferedOutputStream(sock.getOutputStream());

            while (!taskQueue.isEmpty() || (!protocol.shouldTerminate() && connected && (read = in.read()) >= 0) ) {
                if(!taskQueue.isEmpty()){
                    try{
                        while(taskQueue.isEmpty()){
                            semaphore.acquire();
                            send(taskQueue.poll());
                            semaphore.release();
                        }
                    }catch(InterruptedException e){
                        e.printStackTrace();
                    }

                }
                if(read >= 0){
                    T nextMessage = encdec.decodeNextByte((byte) read);
                    if (nextMessage != null) {
                        protocol.process(nextMessage);
                    }
                }
            }
        } catch (IOException ex) {
            ex.printStackTrace();
        }

        connections.disconnect(connectionId);
        try{
            close();
        }catch(IOException e){
            e.printStackTrace();
        }

    }

    @Override
    public void close() throws IOException {
        connected = false;
        sock.close();
    }

    @Override
    public void send(T msg) {
        if (out != null) {
            try {
                out.write(encdec.encode(msg));
                out.flush();
            } catch (IOException ex) {
                ex.printStackTrace();
            }
        }
    }

    public void submit(T msg) {
        if(semaphore.tryAcquire()){
            send(msg);
            semaphore.release();
        }
        else{
            taskQueue.add(msg);
        }
    }
}