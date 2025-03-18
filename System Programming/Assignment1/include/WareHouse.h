#pragma once
#include <string>
#include <vector>

#include "Order.h"
#include "Customer.h"
#include "Volunteer.h"

class BaseAction;
class Volunteer;

// Warehouse responsible for Volunteers, Customers Actions, and Orders.


class WareHouse {

    public:
        WareHouse(const string &configFilePath);
        void start();
        void addOrder(Order* order);
        void addAction(BaseAction* action);
        Customer &getCustomer(int customerId) const;
        Volunteer &getVolunteer(int volunteerId) const;
        Order &getOrder(int orderId) const;
        void increaseOrderCounter();
        const vector<BaseAction*> &getActions() const;
        const int getOrderCounter();//new 22/1
        vector<Order*> &getPendingOrders(); //new 22/1
        vector<Order*> &getInProcessOrders(); //new 22/1
        vector<Order*> &getCompletedOrders();
        vector<Volunteer*> &getVolunteers(); //new 22/1
        vector<Customer*> &getCustomers(); //new 22/1
        int getCustomerCounter();
        int getVolunteerCounter();
        void close();
        void open();

        ~WareHouse();
        WareHouse(const WareHouse& other);
        WareHouse& operator=(const WareHouse& other);
        WareHouse(WareHouse&& other);
        WareHouse& operator=(WareHouse&& other);

    private:
        bool isOpen;
        vector<BaseAction*> actionsLog;
        vector<Volunteer*> volunteers;
        vector<Order*> pendingOrders;
        vector<Order*> inProcessOrders;
        vector<Order*> completedOrders;
        vector<Customer*> customers;
        SoldierCustomer* defaultCustomer = new SoldierCustomer(-1, "", -1, -1);
        CollectorVolunteer* defaultVolunteer = new CollectorVolunteer(-1, "", -1);
        Order* defaultOrder = new Order(-1, -1, -1);
        int customerCounter; //For assigning unique customer IDs
        int volunteerCounter; //For assigning unique volunteer IDs
        int orderCounter;//new 22/1
};