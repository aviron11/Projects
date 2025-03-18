all: bin/warehouse

bin/warehouse: bin/Action.o bin/Customer.o bin/Order.o bin/Volunteer.o bin/WareHouse.o bin/main.o
	@g++ -o bin/warehouse bin/Action.o bin/Customer.o bin/Order.o bin/Volunteer.o bin/WareHouse.o bin/main.o

bin/Action.o:
	@g++ -g -Wall -Weffc++ -std=c++11 -Iinclide -c -o bin/Action.o src/Action.cpp

bin/Customer.o:
	@g++ -g -Wall -Weffc++ -std=c++11 -Iinclide -c -o bin/Customer.o src/Customer.cpp

bin/Order.o:
	@g++ -g -Wall -Weffc++ -std=c++11 -Iinclide -c -o bin/Order.o src/Order.cpp

bin/Volunteer.o:
	@g++ -g -Wall -Weffc++ -std=c++11 -Iinclide -c -o bin/Volunteer.o src/Volunteer.cpp

bin/WareHouse.o:
	@g++ -g -Wall -Weffc++ -std=c++11 -Iinclide -c -o bin/WareHouse.o src/WareHouse.cpp	

bin/main.o:
	@g++ -g -Wall -Weffc++ -std=c++11 -Iinclide -c -o bin/main.o src/main.cpp	

clean:
	@rm -f bin/*