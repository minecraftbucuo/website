# MySQL数据库备份与恢复命令汇总
## 一、备份与恢复命令
1. 备份单个数据库
```bash
mysqldump -u 用户名 -p 数据库名 > 备份文件.sql
```
2. 恢复单个数据库
```bash
mysql -u 用户名 -p 数据库名 < 备份文件.sql
```
3. 备份多个数据库
```bash
mysqldump -u 用户名 -p --databases 数据库1 数据库2 > 备份文件.sql
```
4. 恢复多个数据库
```bash
mysql -u 用户名 -p < 备份文件.sql
```
5. 备份所有数据库
```bash
mysqldump -u 用户名 -p --all-databases > 备份文件.sql
```
6. 恢复所有数据库
```bash
mysql -u 用户名 -p < 备份文件.sql
```
7. 只备份表结构（不含数据）
```bash
mysqldump -u 用户名 -p --no-data 数据库名 > 备份文件.sql
```
8. 恢复表结构（不含数据）
```bash
mysql -u 用户名 -p 数据库名 < 备份文件.sql
```
9. 只备份数据（不含表结构）
```bash
mysqldump -u 用户名 -p --no-create-info 数据库名 > 备份文件.sql
```
10.  恢复数据（不含表结构）
```bash
mysql -u 用户名 -p 数据库名 < 备份文件.sql
```
11.  备份指定表
```bash
mysqldump -u 用户名 -p 数据库名 表名1 表名2 > 备份文件.sql
```
12.  恢复指定表
```bash
mysql -u 用户名 -p 数据库名 < 备份文件.sql
```
13.  生产环境推荐参数（适用于InnoDB，保证数据一致性）
```bash
mysqldump -u 用户名 -p --single-transaction --routines --events 数据库名 > 备份文件.sql
```
14.  恢复生产环境备份
```bash
mysql -u 用户名 -p 数据库名 < 备份文件.sql
```
## 二、其他备份方式
- 物理备份：直接复制MySQL数据目录（如/var/lib/mysql），需停止服务以保证一致性。
- XtraBackup：适合大规模数据库的物理备份工具，支持在线热备份。
- mysqlpump：MySQL 5.7+提供的并行备份工具，效率更高。
## 三、注意事项
- 确保执行用户有足够的数据库权限（如SELECT、SHOW VIEW等）。
- 建议定期通过crontab设置自动化备份任务。
- 备份文件应存储在安全位置，建议异地保存。
- 定期测试恢复流程，确保备份可用性。
- 对于中小规模数据库（30GB以内），mysqldump是最简单有效的方案；数据量较大或对性能要求高时，可考虑XtraBackup等专业工具。