#include "mpi.h"


class com_request{}; // Placeholder


class com_port{
    // Interface to send/receive data

    // Haven't yet decided on how to implement this
    // We can probably get away with only implementing these
    // to work with one type (the one that describes a collection of k-dimensional points)


    // Blocking receive
    template <typename T>
    void receive(T& obj, int sender_rank);

    // Non-blocking receive
    template <typename T>
    com_request receive_begin(T& obj, int sender_rank);
    void receive_wait(com_request request)

    // Blocking send
    template <typename T>
    void send(T& obj, int receiver_rank)


    // Non-blocking send
    template <typename T>
    com_request send_begin(T& obj, int receiver_rank)
    void send_wait(com_request request)
};
