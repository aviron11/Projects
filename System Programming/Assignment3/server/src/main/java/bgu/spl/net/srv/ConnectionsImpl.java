package bgu.spl.net.srv;

import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.LinkedBlockingDeque;
import java.util.concurrent.LinkedBlockingQueue;

public class ConnectionsImpl<T> implements Connections<T> {

    private ConcurrentHashMap<Integer, BlockingConnectionHandler<T>> connections = new ConcurrentHashMap<Integer, BlockingConnectionHandler<T>>();
    private ConcurrentHashMap<Integer, String> idToUsername = new ConcurrentHashMap<Integer, String>();

    public void connect(int connectionId, BlockingConnectionHandler<T> handler){
        connections.put(connectionId, handler);
    }

    public void send(int connectionId, T msg){
        connections.get(connectionId).submit(msg);
    }

    public void disconnect(int connectionId){
        connections.remove(connectionId);
        idToUsername.remove(connectionId);
    }

    public void addUser(int id, String name){
        if (!userExists(name))
            idToUsername.put(id, name);
    }

    public boolean userExists(String name){
        for (Map.Entry<Integer, String> entry : idToUsername.entrySet()) {
            String value = entry.getValue();
            if (value.equals(name))
                return true;
        }
        return false;
    }

    public void removeUser(Integer id){
        idToUsername.remove(id);
    }

}