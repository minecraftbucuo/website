# Arch Linux 极简纯净安装指南 (Btrfs + GRUB)（2026.4.28）

**适用场景：**
* 追求极致纯净，拒绝自动化脚本（如 `archinstall`）的黑盒操作。
* 采用 UEFI 引导。
* 根文件系统采用 Btrfs（支持透明压缩）。
* 目标驱动器示例：`/dev/nvme1n1`

## 预备阶段：连接网络
确保你已经通过 Arch Linux Live U 盘启动。
测试网络连接：
```bash
ping -c 3 baidu.com
```
> **详解：** `ping` 用于测试网络连通性。`-c 3` 参数表示只发送 3 个数据包（count = 3），测完即止，避免它无限期地 ping 下去。如果有回显（Reply），说明网络已经通了，可以开始下载。

---

## 第一阶段：彻底格式化

*警告：请务必确认你的分区号（用 **lsblk** 命令查看），切勿格式化错误的数据盘。本教程以 `p3` 为 EFI 引导分区，`p4` 为系统根分区为例。*

**1. 格式化 EFI 分区为 FAT32**
将指定的引导分区格式化为所有 UEFI 主板都能识别的 FAT32 格式：
```bash
mkfs.fat -F 32 /dev/nvme1n1p3
```
> **详解：** `mkfs` (make file system) 是 Linux 创建文件系统的核心指令。`.fat` 指明了具体的格式。`-F 32` 是强制要求将其格式化为标准的 FAT32 格式。**主板的 UEFI 固件在开机时只认识 FAT 格式的分区**，这就是为什么引导分区必须是它的原因。

**2. 格式化根分区为 Btrfs**
将分配给系统的空间强制格式化为 Btrfs：
```bash
mkfs.btrfs -f /dev/nvme1n1p4
```
> **详解：** `mkfs.btrfs` 调用 Btrfs 专用的格式化工具。`-f` (force) 是强制执行参数，意思是无论这个分区以前装过什么系统或文件，直接无情抹除并覆盖。

---

## 第二阶段：切分 Btrfs 子卷

为了实现精细化管理（例如日后只对系统进行快照而不影响个人文件），我们要在 Btrfs 内部创建子卷。这里我们采用最极简的方案，只区分根目录和用户目录。

**1. 临时挂载顶级卷**
```bash
mount /dev/nvme1n1p4 /mnt
```
> **详解：** `mount` 将物理硬盘分区连接到当前 U 盘系统的一个文件夹（`/mnt`）里。在不指定任何子卷的情况下挂载，我们进入的就是这个 Btrfs 分区的“最顶层（顶级卷）”。只有在最顶层，我们才能去“切分”它。

**2. 创建核心子卷**
```bash
btrfs subvolume create /mnt/@
btrfs subvolume create /mnt/@home
```
> **详解：** `btrfs subvolume create` 是 Btrfs 独有的命令，用于划分逻辑上独立的“子卷”。`@` 和 `@home` 是 Linux 社区约定俗成的命名规范，`@` 用来装系统核心文件，`@home` 用来装用户的个人数据（视频、文档、代码等）。将它们分开，以后系统滚挂了要恢复快照，你的个人文件就不会跟着时光倒流而丢失。

**3. 卸载顶级卷**
```bash
umount /mnt
```
> **详解：** `umount` （注意不是 unmount）用于解除挂载。由于子卷已经建好了，顶级卷的临时使命结束，必须先把它卸载掉，为下一步将各个子卷精准挂载到正确目录做准备。

---

## 第三阶段：正式挂载

这一步我们将子卷以最优参数挂载，开启 `zstd` 透明压缩以节省空间并提升 SSD 寿命。

**1. 挂载系统根目录 (`@`)**
```bash
mount -o noatime,compress=zstd,subvol=@ /dev/nvme1n1p4 /mnt
```
> **详解：** 这是一条优化挂载命令。`-o` 后面跟挂载参数：
> * `noatime`：告诉系统“不要记录文件每次被访问的时间”，大幅减少固态硬盘的无意义写入。
> * `compress=zstd`：开启实时透明压缩，写入硬盘时自动压缩，读取时自动解压，既省空间又变相提升了 I/O 速度。
> * `subvol=@`：精准制导，明确指定这次只挂载名叫 `@` 的这个子系统卷，并将它挂载为整个系统未来的根目录 `/mnt`。

**2. 创建挂载点目录**
```bash
mkdir -p /mnt/home
mkdir -p /mnt/boot/efi
```
> **详解：** `mkdir` (make directory) 用于新建文件夹。`-p` (parents) 是个非常聪明的参数，它会连同缺失的父目录一起创建（比如它会先建 boot 再建 efi），且如果目录已存在也不会报错。这步是在根目录里为接下来的挂载“挖坑”。

**3. 挂载用户目录 (`@home`)**
```bash
mount -o noatime,compress=zstd,subvol=@home /dev/nvme1n1p4 /mnt/home
```
> **详解：** 逻辑同上，将专门存放用户数据的 `@home` 子卷，对准挂载到刚才挖好的 `/mnt/home` 目录坑位里。同样继承了 `noatime` 和 `zstd` 压缩优化。

**4. 挂载 EFI 引导分区**
注意：FAT32 分区不需要（也不支持）Btrfs 的压缩参数。
```bash
mount /dev/nvme1n1p3 /mnt/boot/efi
```
> **详解：** 将最开始格式化的那个 FAT32 引导分区，朴实无华地挂载到 `/mnt/boot/efi` 目录。这是引导程序（GRUB）存放开机文件的地方。

---

## 第四阶段：安装基础系统 (Pacstrap)

**1. 自动测速并优化镜像源**
为了获得最快的下载速度：
```bash
reflector --latest 20 --protocol https --sort rate --save /etc/pacman.d/mirrorlist
```
> **详解：** `reflector` 帮你省去了手动找下载源的麻烦。它会拉取最新同步的 20 个镜像站（`--latest 20`），只选安全的 HTTPS 协议（`--protocol https`），按下载速度从快到慢排序（`--sort rate`），最后将这套最优解直接覆盖保存到系统的下载配置文件中（`--save ...`）。

**2. 极简写入核心包**
向硬盘灌入最精简的内核、微代码、文件系统工具和网络管理器。这里我们选择了高性能的 Zen 内核以及轻量级的 vim 作为系统默认编辑器：
```bash
pacstrap -K /mnt base linux-zen linux-firmware intel-ucode btrfs-progs grub efibootmgr networkmanager vim
```
> **详解：** `pacstrap` 是 Arch 特有的“装机神器”，负责把所有软件强行塞进你刚刚挂载好的新硬盘 `/mnt` 里。`-K` 参数非常重要，它负责初始化并传递 pacman 的安全密钥环，防止安装时报数字签名错误。后面的长串就是你的“基础大礼包”：系统底座、特定内核、Intel 专属微代码、Btrfs 驱动、GRUB 开机套件、网络工具以及 vim。

**3. 生成挂载表 (fstab)**
让系统记住刚才的挂载配置：
```bash
genfstab -U /mnt >> /mnt/etc/fstab
```
> **详解：** `genfstab` (generate file system table) 会扫描你刚才辛辛苦苦敲的一系列 `mount` 操作。`-U` 表示使用分区的 UUID（一长串唯一硬件识别码）来标记硬盘，这比用 `/dev/nvme...` 安全得多（防止以后加减硬盘导致盘符错乱开不了机）。`>>` 将扫描生成的挂载表直接追加写入到新系统的 `/etc/fstab` 文件里。没有这个文件，系统重启后就会找不到硬盘。

---

## 第五阶段：进入系统与引导配置

**1. 切换根目录 (Chroot)**
正式进入你刚刚安装的硬盘系统环境：
```bash
arch-chroot /mnt
```
> **详解：** `chroot` (change root) 堪称系统级的“灵魂穿越”。它将你的终端环境从 U 盘临时系统，瞬间切换到了刚刚装好基础包的真实硬盘里。执行这句之后，你敲击的每一个命令，都是在对你未来的新系统发号施令了。

**2. 开启网络自启**
确保下次重启后自动连接网络：
```bash
systemctl enable NetworkManager
```
> **详解：** `systemctl` 是控制系统后台服务（守护进程）的大管家。`enable` 的意思是“开机自启”。`NetworkManager` 是网络管理服务。如果你漏了这一步，重启进入新系统后，就算插着网线也会没有网络。

**3. 设置 Root 密码**
```bash
passwd
```
> **详解：** 设置最高权限管理员（root）的密码。因为你在 chroot 环境下默认就是 root 身份。注意：在 Linux 终端输密码时，出于安全考虑，屏幕上**绝对不会出现任何星号或占位符**，盲打两遍回车即可。

**4. 部署 GRUB 引导程序**
将 GRUB 安装到 EFI 分区：
```bash
grub-install --target=x86_64-efi --efi-directory=/boot/efi --bootloader-id=Arch
```
> **详解：** 这是真正往主板的启动项里写东西的一步。`--target` 指定平台为 64位 UEFI。`--efi-directory` 告诉它我们刚才把 FAT32 分区挂载在了哪里，它好把引导文件丢进去。`--bootloader-id=Arch` 是给主板 BIOS 看的名字，以后你进 BIOS 调整启动顺序时，看到的就是 "Arch" 这个名字。

**5. 生成 GRUB 菜单**
扫描内核并生成启动项：
```bash
grub-mkconfig -o /boot/grub/grub.cfg
```
*(必须确认屏幕输出了 `Found linux image: /boot/vmlinuz-linux-zen` 等字样。)*
> **详解：** `grub-mkconfig` 是负责“探路”的脚本。它会全面扫描 `/boot` 目录，找出你刚才安装的内核和微代码，并将它们整合成一份机器能读懂的启动配置文件。`-o` (output) 指定了最终输出的文件位置。如果没这一步，GRUB 开机时就会因为找不到内核而陷入黑屏命令行。

---

## 退出与重启

一切就绪！执行以下命令退出安装环境并重启，迎接你的极简系统：
```bash
exit
umount -R /mnt
reboot
```
> **详解：**
> * `exit`：退出灵魂穿越（chroot），回到 U 盘系统。
> * `umount -R /mnt`：`-R` (recursive) 是递归卸载。它会聪明地按照正确的顺序，把包含在 `/mnt` 里面的 `/home`、`/boot/efi` 一层一层安全地拔出来，确保所有数据都已写入硬盘缓存。
> * `reboot`：重启计算机。拔掉 U 盘，开始享受你的成果！