#include "../include/WareHouse.h"
#include "../include/Customer.h"
#include "../include/Volunteer.h"
#include "../include/Action.h"
#include <iostream>
#include <fstream>
#include <sstream>

using namespace std;

WareHouse::WareHouse(const string &configFilePath) :  isOpen(false), actionsLog(), volunteers(), pendingOrders(),
inProcessOrders(), completedOrders(), customers(), customerCounter(0), volunteerCounter(0), orderCounter(0)
{
    
    ifstream inputFile(configFilePath);
    string line;

    while (getline(inputFile, line)) {
        if (line.empty()) {
            continue;
        }
        istringstream iss(line);

        string parsed_position;
        string parsed_name;
        string parsed_type;

        iss >> parsed_position >> parsed_name >> parsed_type;
        if (parsed_position == "customer") {
            int parsed_maxOrders;
            int parsed_locationDistance;
            iss >> parsed_locationDistance >> parsed_maxOrders;
            if (parsed_type == "soldier") {
                SoldierCustomer* cus = new SoldierCustomer(customerCounter, parsed_name, parsed_locationDistance, parsed_maxOrders);
                customers.push_back(cus);
            }
            else {
                CivilianCustomer* cus = new CivilianCustomer(customerCounter, parsed_name, parsed_locationDistance, parsed_maxOrders);
                customers.push_back(cus);
            }
            customerCounter++;
        }

        else if (parsed_position == "volunteer") {
            int parsed_volunteer_coolDown;
            if (parsed_type == "collector") {
                iss >> parsed_volunteer_coolDown;
                CollectorVolunteer* vol = new CollectorVolunteer(volunteerCounter, parsed_name, parsed_volunteer_coolDown);
                volunteers.push_back(vol);
            }
            else if (parsed_type == "limited_collector") {
                int parsed_max_orders;
                iss >> parsed_volunteer_coolDown >> parsed_max_orders;
                LimitedCollectorVolunteer* vol = new LimitedCollectorVolunteer(volunteerCounter, parsed_name, parsed_volunteer_coolDown, parsed_max_orders);
                volunteers.push_back(vol);
            }
            else if (parsed_type == "driver") {
                int parsed_distancePerStep;
                iss >> parsed_volunteer_coolDown >> parsed_distancePerStep;
                DriverVolunteer* vol = new DriverVolunteer(volunteerCounter, parsed_name, parsed_volunteer_coolDown, parsed_distancePerStep);
                volunteers.push_back(vol);
            }
            else if (parsed_type == "limited_driver") {
                int parsed_distancePerStep;
                int parsed_max_orders;
                iss >> parsed_volunteer_coolDown >> parsed_distancePerStep >> parsed_max_orders;
                LimitedDriverVolunteer* vol = new LimitedDriverVolunteer(volunteerCounter, parsed_name, parsed_volunteer_coolDown, parsed_distancePerStep, parsed_max_orders);
                volunteers.push_back(vol);
            }
            volunteerCounter ++;
        }
    }
    inputFile.close();
}

void WareHouse::start() {
    isOpen = true;
    cout << "Warehouse is open!" << endl;
    while(isOpen) {

        string firstWord;
        
        std::cin >> firstWord;
        if (firstWord == "step") {
            int numOfSteps;
            std::cin >> numOfSteps;
            SimulateStep temp = SimulateStep(numOfSteps);
            temp.act(*this);
            addAction(temp.clone());
        }
        else if (firstWord == "order") {
            int customer_id;
            std::cin >> customer_id;
            AddOrder temp = AddOrder(customer_id);
            temp.act(*this);
            addAction(temp.clone());
        }
        else if (firstWord == "customer") {
            string customer_name, customer_type;
            int customer_distance, max_orders;
            std::cin >> customer_name >> customer_type >> customer_distance >> max_orders;
            AddCustomer temp = AddCustomer(customer_name, customer_type, customer_distance, max_orders);
            temp.act(*this);
            addAction(temp.clone());
        }
        else if (firstWord == "orderStatus") {
            int order_id;
            std::cin >> order_id;
            PrintOrderStatus temp = PrintOrderStatus(order_id);
            temp.act(*this);
            addAction(temp.clone());
        }
        else if (firstWord == "customerStatus") {
            int customer_id;
            std::cin >> customer_id;
            PrintCustomerStatus temp = PrintCustomerStatus(customer_id);
            temp.act(*this);
            addAction(temp.clone());
        }
        else if (firstWord == "volunteerStatus") {
            int volunteer_id;
            std::cin >> volunteer_id;
            PrintVolunteerStatus temp = PrintVolunteerStatus(volunteer_id);
            temp.act(*this);
            addAction(temp.clone());
        }
        else if (firstWord == "log") {
            PrintActionsLog temp = PrintActionsLog();
            temp.act(*this);
            addAction(temp.clone());
        }
        else if (firstWord == "close") {
            Close temp = Close();
            temp.act(*this);
            addAction(temp.clone());
            close();
        }
        else if (firstWord == "backup") {
            BackupWareHouse temp = BackupWareHouse();
            temp.act(*this);
            addAction(temp.clone());
        }
        else if (firstWord == "restore") {
            RestoreWareHouse temp = RestoreWareHouse();
            temp.act(*this);
            addAction(temp.clone());
        }
        else {continue;}
    }
}

void WareHouse::addOrder(Order* order) {
    pendingOrders.push_back(order);
}

void WareHouse::addAction(BaseAction* action) {
    actionsLog.push_back(action);
}

Customer &WareHouse::getCustomer(int customerId) const {
    for (Customer* cus : customers) {
        if (cus->getId() == customerId) {
            return *cus;
        }
    }
    return *defaultCustomer;
}

Volunteer &WareHouse::getVolunteer(int volunteerId) const {
    for (Volunteer* vol : volunteers) {
        if (vol->getId() == volunteerId) {
            return *vol;
        }
    }
    return *defaultVolunteer;
}

Order &WareHouse::getOrder(int orderId) const {
    bool found = false;
    for (Order* Order : pendingOrders) {
        if (Order->getId() == orderId) {
            return *Order;
        }
    }
    if(!found) {
        for (Order* Order : inProcessOrders) {
            if (Order->getId() == orderId) {
                return *Order;
            }
        }
    }
    if(!found) {
        for (Order* Order : completedOrders) {
            if (Order->getId() == orderId) {
                return *Order;
            }
        }
    }
    return *defaultOrder;
}

const vector<BaseAction*> &WareHouse::getActions() const {
    return actionsLog;
}

const int WareHouse::getOrderCounter() {
    return orderCounter;
}

vector<Order*> &WareHouse::getPendingOrders() {
    return pendingOrders;
}

vector<Order*> &WareHouse::getInProcessOrders() {
    return inProcessOrders;
}

vector<Order*> &WareHouse::getCompletedOrders() {
    return completedOrders;
}

vector<Volunteer*> &WareHouse::getVolunteers() {
    return volunteers;
}

vector<Customer*> &WareHouse::getCustomers() {
    return customers;
}

int WareHouse::getCustomerCounter() {
    return customerCounter;
}

int WareHouse::getVolunteerCounter() {
    return volunteerCounter;
}

void WareHouse::close() {
    isOpen = false;
}

void WareHouse::open() {
    isOpen = true;
}

void WareHouse::increaseOrderCounter() {
    orderCounter++;
}

WareHouse::~WareHouse() {
    for (Volunteer* vol : volunteers) {
        delete vol;}
    for (Customer* cus : customers) {
        delete cus;}
    for (BaseAction* action : actionsLog) {
        delete action;}
    for (Order* order : pendingOrders) {
        delete order;}
    for (Order* order : inProcessOrders) {
        delete order;}
    for (Order* order : completedOrders) {
        delete order;}
    delete defaultCustomer;
    delete defaultOrder;
    delete defaultVolunteer;
}

WareHouse::WareHouse(const WareHouse& other)
    : isOpen(other.isOpen),
      actionsLog(), volunteers(), 
      pendingOrders(), inProcessOrders(),
      completedOrders(), customers(),
      customerCounter(other.customerCounter),
      volunteerCounter(other.volunteerCounter),
      orderCounter(other.orderCounter) {

    // Perform deep copy of dynamic resources

    for (BaseAction* action : other.actionsLog) {
        actionsLog.push_back(action->clone());
    }

    for (Volunteer* volunteer : other.volunteers) {
        volunteers.push_back(volunteer->clone());
    }

    for (Order* order : other.pendingOrders) {
        pendingOrders.push_back(new Order(*order));
    }

    for (Order* order : other.inProcessOrders) {
        inProcessOrders.push_back(new Order(*order));
    }

    for (Order* order : other.completedOrders) {
        completedOrders.push_back(new Order(*order));
    }

    for (Customer* customer : other.customers) {
        customers.push_back(customer->clone());
    }
}

// Move Constructor
WareHouse::WareHouse(WareHouse&& other)
    : isOpen(other.isOpen),
      actionsLog(std::move(other.actionsLog)),
      volunteers(std::move(other.volunteers)),
      pendingOrders(std::move(other.pendingOrders)),
      inProcessOrders(std::move(other.inProcessOrders)),
      completedOrders(std::move(other.completedOrders)),
      customers(std::move(other.customers)),
      customerCounter(other.customerCounter),
      volunteerCounter(other.volunteerCounter),
      orderCounter(other.orderCounter) {
    // Reset the source object
    other.isOpen = false;
    other.customerCounter = 0;
    other.volunteerCounter = 0;
    other.orderCounter = 0;
}

// Copy Assignment Operator
WareHouse& WareHouse::operator=(const WareHouse& other) {
    if (this != &other) {

    for (Volunteer* vol : volunteers) {
        delete vol;}
    volunteers.clear();
    for (Customer* cus : customers) {
        delete cus;}
    customers.clear();
    for (BaseAction* action : actionsLog) {
        delete action;}
    actionsLog.clear();
    for (Order* order : pendingOrders) {
        delete order;}
    pendingOrders.clear();
    for (Order* order : inProcessOrders) {
        delete order;}
    inProcessOrders.clear();
    for (Order* order : completedOrders) {
        delete order;}
    completedOrders.clear();


        // Copy from other
        isOpen = other.isOpen;
        customerCounter = other.customerCounter;
        volunteerCounter = other.volunteerCounter;
        orderCounter = other.orderCounter;


        // Copy actionsLog
        for (BaseAction* action : other.actionsLog) {
            actionsLog.push_back(action->clone());
        }

        // Copy volunteers
        for (Volunteer* volunteer : other.volunteers) {
            volunteers.push_back(volunteer->clone());
        }

        // Copy orders
        for (Order* order : other.pendingOrders) {
            pendingOrders.push_back(new Order(*order));
        }

        for (Order* order : other.inProcessOrders) {
            inProcessOrders.push_back(new Order(*order));
        }

        for (Order* order : other.completedOrders) {
            completedOrders.push_back(new Order(*order));
        }

        // Copy customers
        for (Customer* customer : other.customers) {
            customers.push_back(customer->clone());
        }
    }
    return *this;
}

// Move Assignment Operator
WareHouse& WareHouse::operator=(WareHouse&& other) {
    if (this != &other) {
        // Release existing resources
        this->~WareHouse();
        // Move from other
        isOpen = other.isOpen;
        actionsLog = std::move(other.actionsLog);
        volunteers = std::move(other.volunteers);
        pendingOrders = std::move(other.pendingOrders);
        inProcessOrders = std::move(other.inProcessOrders);
        completedOrders = std::move(other.completedOrders);
        customers = std::move(other.customers);
        customerCounter = other.customerCounter;
        volunteerCounter = other.volunteerCounter;
        orderCounter = other.orderCounter;

        // Reset the source object
        other.isOpen = false;
        other.customerCounter = 0;
        other.volunteerCounter = 0;
        other.orderCounter = 0;
    }
    return *this;
}