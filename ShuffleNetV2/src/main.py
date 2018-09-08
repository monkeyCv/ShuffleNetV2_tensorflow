# -*- coding: utf-8 -*-
"""
Created on Tue Aug 21 20:32:19 2018

@author: Yuxi1989
"""
import tensorflow as tf
from model import model
import os
import random
    
def load_data():
    #dir param
    train_dir=os.path.join('..','data','train')
    test_dir=os.path.join('..','data','test')
    val_dir=os.path.join('..','data','val')
    
    train_imgs=[]
    train_labels=[]
    val_imgs=[]
    val_labels=[]
    test_imgs=[]
    label_correspond={}
    
    for i,label in enumerate(os.listdir(train_dir)):
        label_correspond[label]=i
        img_dir=os.path.join(train_dir,label,'images')
        for img in os.listdir(img_dir):
            train_imgs.append(os.path.join(img_dir,img))
            train_labels.append(i)
        
    with open(os.path.join(val_dir,'val_annotations.txt')) as file:
        file_labels=[]
        for line in file.readlines():
            strings=line.split('\t')
            file_labels.append(label_correspond[strings[1]])
    for img in sorted(os.listdir(os.path.join(val_dir,'images'))):
        val_imgs.append(os.path.join(val_dir,'images',img))
        img_idx=int(img.split('_')[1].split('.')[0])
        val_labels.append(file_labels[img_idx])
        
    for img in sorted(os.listdir(os.path.join(test_dir,'images'))):
        test_imgs.append(os.path.join(test_dir,'images',img))
    
    train=list(zip(train_imgs,train_labels))
    random.shuffle(train)
    train_imgs[:],train_labels[:]=zip(*train )
    return train_imgs,train_labels,val_imgs,val_labels, test_imgs,label_correspond

def train(train_imgs,train_labels,val_imgs,val_labels):
    has_train=True
    TB_LOG_DIR=os.path.join('..','model')
    ckpt = tf.train.get_checkpoint_state(TB_LOG_DIR)
    if not ckpt and not ckpt.model_checkpoint_path:
        has_train=False
    if has_train==False:
        #dataset param
        EPOCHS=150
        SHUFFLE_SZ=1000
        BATCH_SZ=200
        #model param
        OUTPUT_CNS=[24,48,96,192,1024]
        CLASS_NUM=200
        WEIGHT_DECAY=4e-5
        #training param
        WARM_UP_LR=0.002
        LEARNING_RATE=0.5
        LEARNING_RATE_DECAY=0.95
        TOTAL_STEPS=EPOCHS*100000//BATCH_SZ
        LEARNING_RATE_STEPS=TOTAL_STEPS//100
        MOMENTUM=0.9
        #display
        DISPLAY_STEP=TOTAL_STEPS//100
        TB_LOG_DIR=os.path.join('..','model')
        #validation
        VAL_SZ=10000
    else:
        #dataset param
        EPOCHS=50
        SHUFFLE_SZ=1000
        BATCH_SZ=200
        #model param
        OUTPUT_CNS=[24,48,96,192,1024]
        CLASS_NUM=200
        WEIGHT_DECAY=4e-5
        #training param
        WARM_UP_LR=0.0005  
        LEARNING_RATE=0.0005
        LEARNING_RATE_DECAY=0.9
        TOTAL_STEPS=EPOCHS*100000//BATCH_SZ
        LEARNING_RATE_STEPS=TOTAL_STEPS//100
        MOMENTUM=0.9
        #display
        DISPLAY_STEP=TOTAL_STEPS//100
        TB_LOG_DIR=os.path.join('..','model')
        #validation
        VAL_SZ=10000
        
    
    imgpaths=tf.convert_to_tensor(train_imgs)
    labels=tf.convert_to_tensor(train_labels)
    valimgpaths=tf.convert_to_tensor(val_imgs)
    vallabels=tf.convert_to_tensor(val_labels)
    
    #sess=tf.Session()
    def _parse_function(imgpath,label):
        img=tf.read_file(imgpath)
        img_decoded=tf.image.decode_jpeg(img,3)
        img_decoded.set_shape([64,64,3]) 
        img_decoded=tf.cast(img_decoded,dtype=tf.float32)
        return img_decoded,label    
    dataset=tf.data.Dataset.from_tensor_slices((imgpaths,labels)).map(_parse_function)
    dataset=dataset.shuffle(buffer_size=SHUFFLE_SZ)
    dataset=dataset.repeat(EPOCHS)
    dataset=dataset.batch(BATCH_SZ)
    iterator=dataset.make_initializable_iterator()
    batch_imgs,batch_labels=iterator.get_next()
    
    valset=tf.data.Dataset.from_tensor_slices((valimgpaths,vallabels)).map(_parse_function)
    valset=valset.batch(VAL_SZ)
    valiterator=dataset.make_initializable_iterator()
    valbatch_imgs,valbatch_labels=valiterator.get_next()
    #dimgs,dlabels=sess.run([batch_imgs,batch_labels])
       
    initial=tf.variance_scaling_initializer()
    regular=tf.contrib.layers.l2_regularizer(1.0)
    logits=model(batch_imgs,OUTPUT_CNS,CLASS_NUM,True,regular,initial)
    with tf.name_scope('loss'):
        loss=tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,labels=batch_labels))
        reg=tf.losses.get_regularization_loss()
        loss+=WEIGHT_DECAY*reg
    with tf.name_scope('train'):
        global_step=tf.get_variable('step',shape=[],trainable=False,
                                initializer=tf.zeros_initializer(dtype=tf.int64))
        def get_lr(global_step,total_step,base_lr,warm_up_lr):
            warm_up_total_step=total_step//20
            transition_total_step=warm_up_total_step
            remain_total_step=total_step-warm_up_total_step-transition_total_step
            transition_dlrt=tf.convert_to_tensor((1.0*base_lr-warm_up_lr)/transition_total_step,dtype=tf.float32)
            base_lrt=tf.convert_to_tensor(base_lr,dtype=tf.float32)
            warm_up_lrt=tf.convert_to_tensor(warm_up_lr,dtype=tf.float32)
            warm_up_total_step=tf.convert_to_tensor(warm_up_total_step,dtype=tf.float32)
            transition_total_step=tf.convert_to_tensor(transition_total_step,dtype=tf.float32)
            remain_total_step=tf.convert_to_tensor(remain_total_step,dtype=tf.float32)
            transition_lr=(tf.cast(global_step,tf.float32)-warm_up_total_step)*transition_dlrt+warm_up_lrt
            remain_lr=tf.train.exponential_decay(base_lrt,tf.cast(global_step,tf.float32)-warm_up_total_step-transition_total_step,
                                                  remain_total_step//120 ,LEARNING_RATE_DECAY)
            lr=tf.case({tf.less(global_step,warm_up_total_step): lambda:warm_up_lrt,
                        tf.greater(global_step,transition_total_step+warm_up_total_step): lambda:remain_lr},
                        default=lambda:transition_lr,exclusive=True)
            return lr
        if has_train==False:
            learning_rate=get_lr(global_step,TOTAL_STEPS,LEARNING_RATE,WARM_UP_LR)
        else:
            learning_rate=tf.train.exponential_decay(LEARNING_RATE,global_step,LEARNING_RATE_STEPS,LEARNING_RATE_DECAY) 
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS) 
        with tf.control_dependencies(update_ops): 
            train_op=tf.train.MomentumOptimizer(learning_rate=learning_rate,
                                                momentum=MOMENTUM).minimize(loss)
            with tf.control_dependencies([train_op]):
                global_step_update=tf.assign_add(global_step,1)
                
    if has_train==False:                                      
        init=tf.global_variables_initializer()

    with tf.name_scope('batch_train_accuracy'):
        logits_train=model(batch_imgs,OUTPUT_CNS,CLASS_NUM,False,regular,initial)
        correct_pred_train=tf.equal(tf.cast(tf.argmax(logits_train,1),dtype=tf.int32),batch_labels)
        accuracy_train=tf.reduce_mean(tf.cast(correct_pred_train,tf.float32))
    
    with tf.name_scope('val_accuracy'):
        logits_val=model(valbatch_imgs,OUTPUT_CNS,CLASS_NUM,False,regular,initial)
        correct_pred_val=tf.equal(tf.cast(tf.argmax(logits_val,1),dtype=tf.int32),valbatch_labels)
        accuracy_val=tf.reduce_mean(tf.cast(correct_pred_val,tf.float32))
        
    sess=tf.Session()
    if has_train==False:
        sess.run(init)
    else:
        saver=tf.train.Saver()
        saver.restore(sess,ckpt.model_checkpoint_path)
    sess.run(iterator.initializer)
    
    tf.summary.scalar('loss',loss)
    tf.summary.scalar('batch_train_accuracy',accuracy_train)
    tf.summary.scalar('val_accuracy',accuracy_val)
    tf.summary.scalar('learning_rate',learning_rate)
    tb_merge_summary_op=tf.summary.merge_all()
    summary_writer=tf.summary.FileWriter(os.path.join(TB_LOG_DIR,'tensorboard'),graph=sess.graph)
    
    saver=tf.train.Saver()
    
    sess.run(tf.assign(global_step,0.0))
    for step in range(1,TOTAL_STEPS+1):
        try:
            #_,print_step=sess.run(train_op)
            sess.run(global_step_update)
        except tf.errors.OutOfRangeError:
            break
        if step%DISPLAY_STEP==0 or step==1:
            sess.run(valiterator.initializer)
            l,acct,accv,lr,summary_str=sess.run([loss,accuracy_train,accuracy_val,learning_rate,tb_merge_summary_op])
            summary_writer.add_summary(summary_str,step)
            print("epoch {:d} steps {:d}: loss={:.4f}, accuracy_batch_train={:.4f}, accuracy_val={:.4f}, learning_rate={:.5f}".format(
                    step//(TOTAL_STEPS//EPOCHS),step,l,acct,accv,lr))
    
    summary_writer.close()
    saver.save(sess,os.path.join(TB_LOG_DIR,'model_1.ckpt'))
    
def test(test_imgs):
    TB_LOG_DIR=os.path.join('..','model')
    ckpt = tf.train.get_checkpoint_state(TB_LOG_DIR)
    if not ckpt and not ckpt.model_checkpoint_path:
        print("No model! Please train the model first!")
        return
            
    imgpaths=tf.convert_to_tensor(test_imgs)
    OUTPUT_CNS=[24,48,96,192,1024]
    CLASS_NUM=200
    BATCH_SZ=10000
       
    def _parse_function(imgpath):
        img=tf.read_file(imgpath)
        img_decoded=tf.image.decode_jpeg(img,3)
        img_decoded.set_shape([64,64,3])
        img_decoded=tf.cast(img_decoded,dtype=tf.float32)
        return img_decoded    
    dataset=tf.data.Dataset.from_tensor_slices(imgpaths).map(_parse_function)
    dataset=dataset.batch(BATCH_SZ)
    iterator=dataset.make_one_shot_iterator()
    batch_imgs=iterator.get_next()
    initial=tf.variance_scaling_initializer()
    regular=tf.contrib.layers.l2_regularizer(1.0)
    model(batch_imgs,OUTPUT_CNS,CLASS_NUM,True,regular,initial)
    logits_test=model(batch_imgs,OUTPUT_CNS,CLASS_NUM,False,regular,initial)
    pred=tf.cast(tf.argmax(logits_test,1),dtype=tf.int32)    
    
    sess=tf.Session()
    saver=tf.train.Saver()
    saver.restore(sess,ckpt.model_checkpoint_path)
    prediction=sess.run(pred)
    return prediction

def parse_pred(pred,label_correspond,test_imgs):
    imgs=[x.split('\\')[-1] for x in test_imgs]
    corrs={v:k for k,v in label_correspond.items()}
    labels=[corrs[x] for x in pred]
    num=len(labels)
    with open('../rst/rst.txt','w') as file:
        for idx in range(num):
            file.write('{!s} {!s}\n'.format(imgs[idx],labels[idx]))

if __name__=='__main__':
    tf.reset_default_graph()
    train_imgs,train_labels,val_imgs,val_labels, test_imgs,label_correspond = load_data()
    #train(train_imgs,train_labels,val_imgs,val_labels)
    pred=test(test_imgs)
    parse_pred(pred,label_correspond,test_imgs)
    
        
    
    
        
    