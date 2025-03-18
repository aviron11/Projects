#include "../include/Action.h"
#include "../include/Order.h"
#include "../include/WareHouse.h"
#include "../include/Volunteer.h"
#include <iostream>
#include <algorithm>

using namespace std;

extern WareHouse* backup;

// BaseAction

BaseAction::BaseAction() : errorMsg(""), status(ActionStatus::COMPLETED) {}

ActionStatus BaseAction::getStatus() const {
    return status;
}

void BaseAction::complete() {
    status = ActionStatus::COMPLETED;
}

void BaseAction::error(string errorMsg) {
    status = ActionStatus::ERROR;
    std::cout << errorMsg;
}

string BaseAction::getErrorMsg() const {
    return errorMsg;
}

// SimulateStep

SimulateStep::SimulateStep(int numOfSteps) : numOfSteps(numOfSteps) {}

void SimulateStep::act(WareHouse &wareHouse) {

    
    for (int j=0 ; j < numOfSteps ; j++) {

        int index = 0;
        vector<Order*> &pending = wareHouse.getPendingOrders();
        int pendingSize = (int)pending.size();
        for (int i=0 ; i < pendingSize ; i++) { // Going over pendingOrders
            if(pending.at(i)->getStatus() == OrderStatus::PENDING) {
                for (Volunteer* vol : wareHouse.getVolunteers()) {
                    const string s = vol->getType();
                    if ((s == "CollectorVolunteer" || s == "LimitedCollectorVolunteer") && vol->canTakeOrder(*pending.at(i))) {
                        vol->acceptOrder(*pending.at(i));
                        pending.at(i)->setCollectorId(vol->getId());
                        pending.at(i)->setStatus(OrderStatus::COLLECTING);
                        vol->setActiveOrderId(pending.at(i)->getId());
                        wareHouse.getInProcessOrders().push_back(pending.at(i));
                        wareHouse.getPendingOrders().erase(wareHouse.getPendingOrders().begin()+index);                    
                        index--;
                        i--;
                        pendingSize--;
                        break;
                    }
                }
            }
            else { //collecting
                for (Volunteer* vol : wareHouse.getVolunteers()) {
                    const string s = vol->getType();
                    if ((s == "DriverVolunteer" || s == "LimitedDriverVolunteer") && vol->canTakeOrder(*pending.at(i))) {
                        vol->acceptOrder(*pending.at(i));
                        pending.at(i)->setStatus(OrderStatus::DELIVERING);
                        pending.at(i)->setDriverId(vol->getId());
                        wareHouse.getInProcessOrders().push_back(pending.at(i));
                        wareHouse.getPendingOrders().erase(wareHouse.getPendingOrders().begin()+index);
                        index--;
                        i--;
                        pendingSize--;
                        break;
                    }
                }
            }
            index++;
        }

        index = 0;
        vector<Order*> &inProcess = wareHouse.getInProcessOrders();
        int inProcessSize = inProcess.size();
        for (int i=0 ; i < (int)inProcessSize ; i++) {
            if (inProcess.at(i)->getStatus() == OrderStatus::COLLECTING) {
                int collectorId = inProcess.at(i)->getCollectorId();
                dynamic_cast<CollectorVolunteer&>(wareHouse.getVolunteer(collectorId)).step(); 
                int test = dynamic_cast<CollectorVolunteer&>(wareHouse.getVolunteer(collectorId)).getTimeLeft();
                if (test == 0){
                    (wareHouse.getPendingOrders()).push_back(inProcess.at(i));
                    (wareHouse.getInProcessOrders()).erase(wareHouse.getInProcessOrders().begin()+index);
                    index--;
                    dynamic_cast<CollectorVolunteer&>(wareHouse.getVolunteer(collectorId)).reset();
                    i--;
                    inProcessSize--;
                    (wareHouse.getVolunteer(collectorId)).setActiveOrderId(NO_ORDER);
                }
            }
            else { //delivering 
                int driverId = inProcess.at(i)->getDriverId();
                dynamic_cast<DriverVolunteer&>(wareHouse.getVolunteer(driverId)).step();
                int test = dynamic_cast<DriverVolunteer&>(wareHouse.getVolunteer(driverId)).getDistanceLeft();
                if (test <= 0){
                    inProcess.at(i)->setStatus(OrderStatus::COMPLETED);
                    (wareHouse.getCompletedOrders()).push_back(inProcess.at(i));
                    (wareHouse.getInProcessOrders()).erase(wareHouse.getInProcessOrders().begin()+index);
                    index--;
                    dynamic_cast<DriverVolunteer&>(wareHouse.getVolunteer(driverId)).reset();
                    i--;
                    inProcessSize--;
                    (wareHouse.getVolunteer(driverId)).setActiveOrderId(NO_ORDER);
                }
            }
            index++;
        }

        index = 0;
        vector<Volunteer*> &volunteers = wareHouse.getVolunteers();
        for (int i = 0 ; i < (int)volunteers.size() ; i++) {
            if (volunteers.at(i)->getType() == "LimitedCollectorVolunteer") {
                int test = dynamic_cast<LimitedCollectorVolunteer&>(*volunteers.at(i)).getNumOrdersLeft();
                if (test == 0 && !dynamic_cast<LimitedCollectorVolunteer&>(*volunteers.at(i)).isBusy()) {
                    delete volunteers.at(i);
                    (wareHouse.getVolunteers()).erase(wareHouse.getVolunteers().begin()+index);
                    index--;
                }
            }
            else if (volunteers.at(i)->getType() == "LimitedDriverVolunteer") {
                int test = dynamic_cast<LimitedDriverVolunteer&>(*volunteers.at(i)).getNumOrdersLeft();
                if (test == 0 && !dynamic_cast<LimitedDriverVolunteer&>(*volunteers.at(i)).isBusy()) {
                    delete volunteers.at(i);
                    (wareHouse.getVolunteers()).erase(wareHouse.getVolunteers().begin()+index);
                    index--;
                }
            }
            index++;
        }
    }
}

std::string SimulateStep::toString() const {
    return "simulateStep " + std::to_string(numOfSteps) + " COMPLETED";
}

SimulateStep *SimulateStep::clone() const {
    return new SimulateStep(*this);
}

// AddOrder

AddOrder::AddOrder(int id) : customerId(id) {}

void AddOrder::act(WareHouse &wareHouse) {
    int dist;
    bool test = true;
    for (Customer* cus : wareHouse.getCustomers()) {
        if(cus->getId() == customerId){
            if(cus->canMakeOrder()) {
                dist = cus->getCustomerDistance();
                Order* order = new Order(wareHouse.getOrderCounter(), customerId, dist);
                cus->addOrder(wareHouse.getOrderCounter());
                test = false;
                complete();
                wareHouse.increaseOrderCounter();
                wareHouse.addOrder(order);
                break;
            }
        }
    }
    if (test) {
        error("Error: Cannot place this order \n");
    }
}

AddOrder *AddOrder::clone() const {
    return new AddOrder(*this);
}

string AddOrder::toString() const {
    if (getStatus() == ActionStatus::COMPLETED) {
        return "order " + std::to_string(customerId) + " COMPLETED";
    }
    return "order " + std::to_string(customerId) + " ERROR";
}

// AddCustomer

AddCustomer::AddCustomer(const string &customerName, const string& customerType, int distance, int maxOrders)
    : customerName(customerName), customerType((customerType == "Civilian") ? CustomerType::Civilian : CustomerType::Soldier), distance(distance), maxOrders(maxOrders) {}

void AddCustomer::act(WareHouse &wareHouse) {
    if (customerType == CustomerType::Soldier) {
        SoldierCustomer* cus = new SoldierCustomer(wareHouse.getCustomerCounter(), customerName, distance, maxOrders);
        wareHouse.getCustomers().push_back(cus);
    }
    else {
        CivilianCustomer* cus = new CivilianCustomer(wareHouse.getCustomerCounter(), customerName, distance, maxOrders);
        wareHouse.getCustomers().push_back(cus);
    }
}

AddCustomer *AddCustomer::clone() const {
    return new AddCustomer(*this);
}

string AddCustomer::toString() const {
    return "add customer COMPLETED";
}

// PrintOrderStatus

PrintOrderStatus::PrintOrderStatus(int id) : orderId(id) {}

void PrintOrderStatus::act(WareHouse &wareHouse) {
    bool found = false;
    for (Order* order : wareHouse.getPendingOrders()){
        if (order->getId() == orderId) {
            std::cout << "OrderId: " << orderId << endl;
            std::cout << "OrderStatus: " << order->getStringStatus() << endl;
            std::cout << "CustomerID: " << order->getCustomerId() << endl;
            if (order->getCollectorId() == NO_VOLUNTEER) {
                std::cout << "Collector: None" << endl;
            }
            else {
                std::cout << "Collector: " << order->getCollectorId() << endl;
            }
            std::cout << "Driver: None" << endl;
            found = true;
            complete();
        }
    }
    if (!found) {
        for (Order* order : wareHouse.getInProcessOrders()){
            if (order->getId() == orderId) {
                std::cout << "OrderId: " << orderId << endl;
                std::cout << "OrderStatus: " << order->getStringStatus() << endl;
                std::cout << "CustomerID: " << order->getCustomerId() << endl;
                std::cout << "Collector: " << order->getCollectorId() << endl;
                if (order->getDriverId() == NO_VOLUNTEER) {
                    std::cout << "Driver: None" << endl;
                }
                else {
                    std::cout << "Driver: " << order->getDriverId() << endl;
                }
                found = true;
                complete();
            }
        }
    }
    if (!found) {
        for (Order* order : wareHouse.getCompletedOrders()){
            if (order->getId() == orderId) {
                std::cout << "OrderId: " << orderId << endl;
                std::cout << "OrderStatus: " << order->getStringStatus() << endl;
                std::cout << "CustomerID: " << order->getCustomerId() << endl;
                std::cout << "Collector: " << order->getCollectorId() << endl;
                std::cout << "Driver:" << order->getDriverId() << endl;                
                found = true;
                complete();
            }
        }
    }
    if(!found) {
        error("Order doesn't exist \n");
    }
}

PrintOrderStatus *PrintOrderStatus::clone() const {
    return new PrintOrderStatus(*this);
}

string PrintOrderStatus::toString() const {
    if (this->getStatus() == ActionStatus::COMPLETED) {
        return "orderStatus " + std::to_string(orderId) + " COMPLETED";
    }
    return "orderStatus " + std::to_string(orderId) + " ERROR";
}

// PrintCustomerStatus

PrintCustomerStatus::PrintCustomerStatus(int customerId) : customerId(customerId) {}

void PrintCustomerStatus::act(WareHouse &wareHouse) {
    Customer* cus = &wareHouse.getCustomer(customerId);

    if (cus->getId() == -1) {
        error("Customer doesn't exist \n");
    }
    else {
        std::cout << "CustomerID: " << customerId << endl;
        int size = (int)cus->getOrdersIds().size();
        for(int i=0 ; i < size ; i++) {
            int id = cus->getOrdersIds().at(i);
            std::cout << "OrderID: " << id << endl <<
            "OrderStatus: " << wareHouse.getOrder(id).getStringStatus() << endl;
        }
        std::cout << "numOrdersLeft: " << cus->getMaxOrders() - cus->getNumOrders() << endl;
    }
}

PrintCustomerStatus *PrintCustomerStatus::clone() const {
    return new PrintCustomerStatus(*this);
}

string PrintCustomerStatus::toString() const {
    if (this->getStatus() == ActionStatus::COMPLETED) {
        return "customerStatus " + std::to_string(customerId) + " COMPLETED";
    }
    return "customerStatus " + std::to_string(customerId) + " ERROR";
}

// PrintVolunteerStatus

PrintVolunteerStatus::PrintVolunteerStatus(int id) : volunteerId(id) {}

void PrintVolunteerStatus::act(WareHouse &wareHouse) {
    Volunteer* vol = &wareHouse.getVolunteer(volunteerId);
    string type = vol->getType();
    
    if (vol->getId() == -1) {
        error("Volunteer doesn't exist \n");
    }
    else {
        std::cout << "VolunteerID: " << volunteerId << endl;
        bool busy = vol->isBusy();
        if(busy) {std::cout << "isBusy: True" << endl;}
        else {std::cout << "isBusy: False" << endl; }
        std::cout << "OrderID: ";
        if (!busy) {std::cout << "None" << endl;}
        else {std::cout << vol->getActiveOrderId() << endl;}
        std::cout << "timeLeft: ";
        string volType = vol->getType();
        if (!busy) {std::cout << "None" << endl;}
        else if (volType == "CollectorVolunteer" || volType == "LimitedCollectorVolunteer")
        {
            std::cout << dynamic_cast<CollectorVolunteer*>(vol)->getTimeLeft() << endl;
        }
        else {std::cout << dynamic_cast<DriverVolunteer*>(vol)->getDistanceLeft() << endl;}
        if (volType == "CollectorVolunteer" || volType == "DriverVolunteer") 
        {
            std::cout << "OrdersLeft: No Limit" << endl;
        }
        else if (volType == "LimitedCollectorVolunteer") 
        {std::cout << "OrdersLeft: " << dynamic_cast<LimitedCollectorVolunteer*>(vol)->getNumOrdersLeft() << endl;}
        else if (volType == "LimitedDriverVolunteer") 
        {std::cout << "OrdersLeft: " << dynamic_cast<LimitedDriverVolunteer*>(vol)->getNumOrdersLeft() << endl;}
    }
}

PrintVolunteerStatus *PrintVolunteerStatus::clone() const {
    return new PrintVolunteerStatus(*this);
}

string PrintVolunteerStatus::toString() const {
    if (this->getStatus() == ActionStatus::COMPLETED) {
        return "volunteerStatus " + std::to_string(volunteerId) + " COMPLETED";
    }
    return "volunteerStatus " + std::to_string(volunteerId) + " ERROR";
}

// PrintActionsLog

PrintActionsLog::PrintActionsLog() {}

void PrintActionsLog::act(WareHouse &wareHouse) {
    for (BaseAction* action : wareHouse.getActions()) {
        std::cout << action->toString() << endl;
    }
    complete();
}

PrintActionsLog *PrintActionsLog::clone() const {
    return new PrintActionsLog(*this);
}

string PrintActionsLog::toString() const {
    return "log COMPLETED";
}

// Close

Close::Close() {}

void Close::act(WareHouse &wareHouse) {
    for (Order* order : wareHouse.getPendingOrders()) {
        std::cout << "OrderID: " << order->getId() << ", CustomerID: " 
        << order->getCustomerId() << ", Status: " << order->getStringStatus() << endl;
    }
    for (Order* order : wareHouse.getInProcessOrders()) {
        std::cout << "OrderID: " << order->getId() << ", CustomerID: " 
        << order->getCustomerId() << ", Status: " << order->getStringStatus() << endl;
    }
    for (Order* order : wareHouse.getCompletedOrders()) {
        std::cout << "OrderID: " << order->getId() << ", CustomerID: " 
        << order->getCustomerId() << ", Status: " << order->getStringStatus() << endl;
    }
    complete();
}

Close *Close::clone() const {
    return new Close(*this);
}

string Close::toString() const {
    return "Close COMPLETED";
}

// BackupWareHouse

BackupWareHouse::BackupWareHouse() {}

void BackupWareHouse::act(WareHouse &wareHouse) {
    if (backup != nullptr) {delete backup;}
    backup = new WareHouse(wareHouse);
    complete();
}

BackupWareHouse *BackupWareHouse::clone() const {
    return new BackupWareHouse(*this);
}

string BackupWareHouse::toString() const {
    return "Backup COMPLETED";
}

// RestoreWareHouse   

RestoreWareHouse::RestoreWareHouse() {}

void RestoreWareHouse::act(WareHouse &wareHouse) {
    if (backup == nullptr) {
        error("No backup available \n");
    }
    else {
        wareHouse = *backup;
    }
}

RestoreWareHouse *RestoreWareHouse::clone() const {
    return new RestoreWareHouse(*this);
}

string RestoreWareHouse::toString() const {
    if (this->getStatus() == ActionStatus::COMPLETED) {
        return "Restore COMPLETED";
    }
    return "Restore ERROR";
}
