import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import matplotlib.dates as mdates
date_format = "%d.%m.%Y"


with open("conversation.txt", "r") as file:  # Open the exported WhatsApp message data
    text_data = file.read()
    lines = text_data.split("\n")
    total = []

    # Names of the people that are in the given chat by their saved names there.
    separated = {"user1": [], "user2": [], "user3": [], "user4": []}

    last_person = ""
    # Concatenates messages that should not be separated by the .split("\n") above.
    # Some messages consist of multiple lines. This code brings them together
    # under the correct persons name.
    for i, line in enumerate(lines):

        # You can remove "elif" blocks here if there are less people in the chat.
        # However, nothing happens if you leave it as is.
        # If you have n people in the chat, related sections and variables should
        # include n fields for those. Like "separated" dictionary above.

        if " - user1: " in line:  # Strings here are formatted in WhatsApp's formatting.
            lines[i] = line.replace(" - user1: ", "")
            total.append(lines[i])
            separated["user1"].append(lines[i])
            last_person = "user1"
        elif " - user2: " in line:
            lines[i] = line.replace(" - user2: ", "")
            total.append(lines[i])
            separated["user2"].append(lines[i])
            last_person = "user2"
        elif " - user3: " in line:
            lines[i] = line.replace(" - user3: ", "")
            total.append(lines[i])
            separated["user3"].append(lines[i])
            last_person = "user3"
        elif " - user4: " in line:
            lines[i] = line.replace(" - user4: ", "")
            total.append(lines[i])
            separated["user4"].append(lines[i])
            last_person = "user4"
        elif last_person:
            separated[last_person][-1] += line

    # Get message counts
    user1_count = len(separated["user1"])
    user2_count = len(separated["user2"])
    user3_count = len(separated["user3"])
    user4_count = len(separated["user4"])


    # Get total message lengths
    user1_len = 0
    user2_len = 0
    user3_len = 0
    user4_len = 0
    for message in separated["user1"]:
        user1_len += len(message)
    for message in separated["user2"]:
        user2_len += len(message)
    for message in separated["user3"]:
        user3_len += len(message)
    for message in separated["user4"]:
        user4_len += len(message)


    timestamps = []
    # Timestamps are created from the WhatsApp's text formatting.
    # Timestamps are created day by day, and listed here as uniquely.
    # This code will form the x-axis labels of the plots.
    for i, block in enumerate(total):
        tstring = block[:10]
        if tstring[-1] == " ":
            tstring = "0" + tstring[:-1]
        try:
            timestamps.append(datetime.strptime(tstring, date_format))
        except ValueError:
            pass

count_data = {}

# Collect message counts day-by-day
for stamp in timestamps:
    if stamp in count_data:
        count_data[stamp] += 1
    else:
        count_data[stamp] = 1

dates = list(count_data.keys())
counts = np.array(list(count_data.values()))  # Message counts. Convolution will be applied to those.
                                              # That is why I am turning them into numpy arrays.

# Convolution kernels
kernel1 = np.array([1, 3, 1]) / 5
kernel2 = np.array([1, 2, 3, 2, 1]) / 9
kernel3 = np.array([2, 1, 3, 1, 2]) / 9
kernel4 = np.array([1, 1, 5, 1, 1]) / 9

density1 = np.convolve(counts, kernel1, "valid") / 16
density2 = np.convolve(counts, kernel2, "valid") / 16
density3 = np.convolve(counts, kernel3, "valid") / 16
density4 = np.convolve(counts, kernel4, "valid") / 16

fig = plt.figure(figsize=(15, 30))
fig.suptitle("CHAT NAME")

ax1 = plt.subplot2grid((6, 2), (0, 0), colspan=2)
ax1.set(title='Message Count by Day', xlabel='Date', ylabel='Message Count')
ax1.bar(dates, counts, width=1)
ax1.xaxis.set_major_locator(mdates.DayLocator(interval=28))
ax1.xaxis.set_major_formatter(mdates.DateFormatter(date_format))
ax1.tick_params(axis='x', labelsize=8, rotation=45)

ax2 = plt.subplot2grid((6, 2), (1, 0))
ax2.set(title="Message Count Ratios")
ax2.pie([user1_count, user2_count, user3_count, user4_count], labels=["user1", "user2", "user3", "user4"], autopct='%.2f%%')
# Replace labels here with usernames.

ax3 = plt.subplot2grid((6, 2), (1, 1))
ax3.set(title="Mesaj Uzunluk Oranları")
ax3.pie([user1_len, user2_len, user3_len, user4_len], labels=["user1", "user2", "user3", "user4"], autopct='%.2f%%')

ax4 = plt.subplot2grid((6, 2), (2, 0), colspan=2)
ax4.set(title="Message Density (message/hour) | kernel: [1, 3, 1]")
ax4.bar(dates[1:-1], density1)

ax5 = plt.subplot2grid((6, 2), (3, 0), colspan=2)
ax5.set(title="Message Density (message/hour) | kernel: [1, 2, 3, 2, 1]")
ax5.bar(dates[2:-2], density2)

ax6 = plt.subplot2grid((6, 2), (4, 0), colspan=2)
ax6.set(title="Message Density (message/hour) | kernel: [2, 1, 3, 1, 2]")
ax6.bar(dates[2:-2], density3)

ax7 = plt.subplot2grid((6, 2), (5, 0), colspan=2)
ax7.set(title="Message Density (message/hour) | kernel: [1, 1, 5, 1, 1]")
ax7.bar(dates[2:-2], density4)

plt.subplots_adjust(hspace=0.3, bottom=0.01)
plt.show()
