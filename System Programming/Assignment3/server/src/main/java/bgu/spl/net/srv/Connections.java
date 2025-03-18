package bgu.spl.net.srv;

import java.io.IOException;

public interface Connections<T> {

    void connect(int connectionId, BlockingConnectionHandler<T> handler);

    void send(int connectionId, T msg);

    void disconnect(int connectionId);

    public void addUser(int id, String name);

    public void removeUser(Integer id);

    public boolean userExists(String name);

}
