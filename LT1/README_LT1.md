# Set-up instructions for Raspberry Pi data collection device in LT1

- Note: when I refer to a file in the LT1/ directory, it means that it is to be found in the set of templates that should accompany this README. Therefore to 'Copy LT1/on-boot.sh to /home/pirate' means to take the template from LT1/ and transfer it to the Raspberry Pi first. How you do that is up to you.

- Get Raspberry Pi with SD card for running the core OS. They ship with a 16GB SD card; that should be fine, but I used 32GB to be sure.

- Install hypriot onto the SD card, e.g. insert the SD card into your computer's SD reader, find the device <name> using lsblk and then running dd if=hypriot*img of=/dev/<name> bs=4M status=progress, or one of the many such SD imaging packages for Windows, etc.
  * Edit the boot (1st) partition's 'user-data' file and set the hostname setting if desired to have the hostname configured from first boot. I've named them with the scheme 'lt1-rpi<n>'.
  * It is possible to configure networking from first boot but the UniOfCam-IoT network requires knowing the wifi adapter MAC address, which is hard to obtain without booting the system. Therefore you may have to plug the Pi into a keyboard and monitor in order to get the MAC address and configure networking.
  * You can setup the password for the 'pirate' user at this time or do it later.
    + The default password for HypriotOS is 'hypriot'

- Basics, upon booting for the first time:
  * If you didn't configure the hostname earlier do it now:
    + sudo hostnamectl set-hostname --static <hostname>
    + Edit /etc/cloud/cloud.cfg and change preserve_hostname to true.
  * Set the pirate user's password.
  * echo alias temp=\'vcgencmd measure_temp\' >> ~/.bashrc

- WiFi setup:
  * Create a device on the UniOfCam-IoT website, using the WiFi adapter MAC address that you obtained somehow, probably by booting the Raspberry Pi and writing it down from the output of ifconfig wlan0.
  * Copy LT1/wpa_supplicant.conf to /etc/wpa_supplicant/ and edit the 'psk' setting to the password given by UniOfCam-IoT configuration.
  * Copy LT1/wlan0 to /etc/network/interfaces.d/
  * Run: sudo ifup wlan0, or reboot

- SSH tunnel setup:
  * Run ssh-keygen, and hit enter 3 times to accept defaults and blank passphrase. 
  * Copy LT1/ssh-config info ~/.ssh/config
    + There should be two hosts in the config file, 'tfc-forwards' for the tunnels and 'tfc' for your convenience.
    + Change remote forwarding port numbers to unique ones (current scheme is 1n022 for SSH and 1n081 for the motion camera preview, where n is 1-9). 
    + Change the User fields to match the desired user on tfc.
  * Test the SSH connections to tfc and tfc-forwards to ensure that SSH known-hosts are updated.
  * Put .ssh/id_rsa.pub into tfc's authorized_keys
    + For example, by running this command: cat ~/.ssh/id_rsa.pub | ssh tfc 'cat >> .ssh/authorized_keys'
  * apt update; apt install autossh
  * Edit autossh-tunnel.service
    + Change the -M <port> value to something unique (current scheme is 1n078 where n is 1-9).
    + Confirm that the configuration file has autossh connecting to tfc-forwards and that the ExecPreStart is testing the DNS service (e.g. with tfc-app9.cl.cam.ac.uk).
  * cp autossh-tunnel.service /etc/systemd/system/
  * sudo systemctl enable autossh-tunnel.service
  * useful commands:
    + sudo systemctl restart autossh-tunnel.service
    + sudo systemctl status autossh-tunnel.service

  * At this point it should be possible to reboot the Pi and after letting it boot up and settle down for a few minutes the ssh tunnel should be accessible from tfc-app9 using the port that you configured in .ssh/config.


- USB thumb drive setup:
  * Add this entry to /etc/fstab: /dev/sda1 /store vfat noauto,user 0 0
  * sudo mkdir /store
  * sudo apt install rsync

- Motion setup:
  * docker pull mrdanish/motioneye-rpi
    + This may take a while...
  * mount /store  # As pirate user, NOT as root
  * Copy LT1/motion/motion.conf onto the USB thumb drive under directory 'motion' and edit the conf file to change the camera_name to match desired name (probably the same as the hostname, for simplicity). Check the picture_output and movie_output settings (I have used 'yes' to picture_output and 'no' to movie_output for now).
  * Copy LT1/backup.sh and LT1/clean.sh onto the USB thumb drive and edit the DEFAULTCAM value to match the name of the device (again, probably the hostname).
  * sudo apt install screen
  * Copy LT1/pause.sh LT1/resume.sh LT1/on-boot.sh LT1/run.sh LT1/cron.tab into /home/pirate
  * Set permissions: chmod 0755 /home/pirate/on-boot.sh /home/pirate/run.sh /home/pirate/pause.sh /home/pirate/resume.sh
  * If you want the motion software to run on boot, then run: crontab cron.tab
  * Otherwise you have to run it yourself using run.sh
  * The crontab file starts the motion software inside a screen session, so when you login to the Pi later you can bring the console up by typing 'screen -r'.
  * The crontab file also includes use of the backup and clean scripts, so you should ensure that the USB stick has been set-up as described above, and the tfc server is ready to accept those files, and if not then comment out those lines in the crontab.
  * With the default cron settings, backup occurs at 1 a.m. and clean occurs at 4 a.m. following a successful backup.
    + The clean.sh script checks for successful backup before removing any files.
  * pause.sh and resume.sh are control scripts for pausing or resuming the motion capture process from the shell.
    + For example, if you need to manually intervene and stop the motion capture temporarily.

- Camera setup:
  * Sometimes the camera does not come configured correctly.
  * Run: sudo raspi-config and go into the Interfacing Options to ensure the Camera is enabled.
  * Check /boot/config.txt, make sure it has the following settings:
    + start_x=1
    + disable_camera_led=1
    + gpu_mem=128
  * For more about camera settings, see http://elinux.org/RPiconfig#Camera
  * Copy LT1/led-off.sh and chmod 0755 it if you want a script to turn the camera LED off.

