interfaces=$(ls /sys/class/net)

# Loop through each interface and disable offload features
for interface in $interfaces
do
    echo $interface
    sudo ethtool -K $interface gro off
    sudo ethtool -K $interface tx off
    sudo ethtool -K $interface rx off
    sudo ethtool -K $interface rxvlan off
    sudo ethtool -K $interface txvlan off
    sudo ethtool -K $interface sg off
    sudo ethtool -K $interface tso off
    sudo ethtool -K $interface gso off
    sudo ethtool -K $interface ufo off
    sudo ethtool -K $interface lro off
    sudo ethtool -K $interface rxhash off
done
