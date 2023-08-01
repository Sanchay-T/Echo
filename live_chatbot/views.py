from django.shortcuts import render

# Create your views here.

def index(request):
    return render(request, "index_2.html")


def room(request, room_name):
    return render(request, "room.html", {"room_name": room_name})

